import asyncio
import logging
import os
import re
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from bs4 import BeautifulSoup, Comment, Doctype
from deep_translator import GoogleTranslator
from ebooklib import ITEM_DOCUMENT, epub
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger("epub-bot")


# ============================================================
# CONFIG
# ============================================================

TOKEN = os.environ["BOT_TOKEN"]

MAX_MB = int(os.getenv("MAX_FILE_MB", "20"))

WORKERS = max(
    1,
    min(
        int(os.getenv("TRANSLATE_WORKERS", "3")),
        8,
    ),
)

CHUNK = max(
    1000,
    min(
        int(os.getenv("CHUNK_SIZE", "3500")),
        4500,
    ),
)

RETRIES = max(
    1,
    min(
        int(os.getenv("TRANSLATE_RETRIES", "5")),
        8,
    ),
)

BATCH_CHARS = max(
    CHUNK,
    min(
        int(os.getenv("BATCH_CHARS", "3500")),
        4500,
    ),
)


# ============================================================
# LANGUAGES
# ============================================================

LANGS = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "pt": "Português",
    "ru": "Русский",
    "ar": "العربية",
    "hi": "Hindi",
    "bn": "Bengali",
    "id": "Indonesian",
    "tr": "Türkçe",
    "zh-CN": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "it": "Italiano",
    "vi": "Vietnamese",
}


# Tags whose contents should NOT be translated.
SKIP_TAGS = {
    "script",
    "style",
    "code",
    "pre",
    "noscript",
    "svg",
}


_ws = re.compile(r"[ \t]+")


# ============================================================
# TEXT HELPERS
# ============================================================

def chunk_text(text: str, size: int = CHUNK):
    """
    Split long text into smaller pieces while trying
    to split at whitespace.
    """

    text = text.strip()

    while len(text) > size:

        cut = text.rfind(" ", 0, size)

        if cut < size * 0.55:
            cut = size

        yield text[:cut]

        text = text[cut:].lstrip()

    if text:
        yield text


def restore_ws(original: str, translated: str) -> str:
    """
    Restore leading/trailing whitespace.
    """

    lead = original[
        : len(original) - len(original.lstrip())
    ]

    trail = original[
        len(original.rstrip()):
    ]

    return lead + translated + trail


# ============================================================
# TRANSLATION
# ============================================================

def translate_with_retry(
    translator,
    text: str,
) -> str:

    if not text.strip():
        return text

    parts = []

    for part in chunk_text(text):

        result = None

        for attempt in range(RETRIES):

            try:

                result = translator.translate(part)

                if result:
                    break

            except Exception as exc:

                delay = min(
                    12,
                    1.5 * (2 ** attempt),
                ) + (attempt * 0.15)

                log.warning(
                    "translation retry %d/%d: %s",
                    attempt + 1,
                    RETRIES,
                    exc,
                )

                time.sleep(delay)

        if not result:

            # NEVER lose original text.
            log.error(
                "translation failed; preserving original chunk"
            )

            result = part

        parts.append(result)

    return " ".join(parts)


def translate_batch(batch, target):

    translator = GoogleTranslator(
        source="auto",
        target=target,
    )

    if len(batch) == 1:
        return [
            translate_with_retry(
                translator,
                batch[0],
            )
        ]

    joined = "\n".join(batch)

    for attempt in range(RETRIES):

        try:

            result = translator.translate(joined)

            if result:

                lines = result.splitlines()

                if len(lines) == len(batch):

                    return [
                        x.strip()
                        for x in lines
                    ]

        except Exception as exc:

            delay = min(
                10,
                1.25 * (2 ** attempt),
            )

            log.warning(
                "batch retry %d/%d: %s",
                attempt + 1,
                RETRIES,
                exc,
            )

            time.sleep(delay)

    # Safe fallback.
    return [
        translate_with_retry(
            translator,
            item,
        )
        for item in batch
    ]


def translate_many(
    texts,
    target,
    progress=None,
    workers=WORKERS,
):

    clean = []

    for text in texts:

        if not text.strip():
            clean.append("")

        else:
            clean.append(
                _ws.sub(
                    " ",
                    text.strip(),
                )
            )

    batches = []
    indexes = []

    current = []
    current_idx = []
    chars = 0

    for i, text in enumerate(clean):

        if not text:
            continue

        extra = len(text)

        if current:
            extra += 1

        if (
            current
            and chars + extra > BATCH_CHARS
        ):

            batches.append(current)
            indexes.append(current_idx)

            current = []
            current_idx = []
            chars = 0

        current.append(text)
        current_idx.append(i)
        chars += extra

    if current:
        batches.append(current)
        indexes.append(current_idx)

    out = list(clean)

    total = len(batches)

    if not total:
        return texts

    completed = 0

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {
            executor.submit(
                translate_batch,
                batch,
                target,
            ): n
            for n, batch in enumerate(batches)
        }

        for future in as_completed(futures):

            n = futures[future]

            try:

                result = future.result()

            except Exception:

                log.exception(
                    "batch crashed; preserving original text"
                )

                result = batches[n]

            for i, value in zip(
                indexes[n],
                result,
            ):

                out[i] = value

            completed += 1

            if progress:

                progress(
                    completed,
                    total,
                )

    return [
        restore_ws(
            original,
            translated,
        )
        if original.strip()
        else original

        for original, translated
        in zip(texts, out)
    ]


# ============================================================
# EPUB TOC
# ============================================================

def collect_toc_entries(book):

    entries = []

    def walk(items):

        for item in items:

            if isinstance(item, tuple):

                if item:
                    entries.append(item[0])

                if len(item) > 1:
                    walk(item[1])

            elif hasattr(item, "title"):

                entries.append(item)

    walk(book.toc)

    return entries


# ============================================================
# EPUB TRANSLATION
# ============================================================

def translate_epub(
    src,
    dst,
    target,
    progress=None,
):

    log.info(
        "Opening EPUB: %s",
        src,
    )

    book = epub.read_epub(src)

    docs = [
        item
        for item in book.get_items()
        if item.get_type() == ITEM_DOCUMENT
    ]

    soups = []
    nodes = []

    # --------------------------------------------------------
    # Extract text nodes
    # --------------------------------------------------------

    for item in docs:

        soup = BeautifulSoup(
            item.get_content(),
            "html.parser",
        )

        soups.append(soup)

        for node in soup.find_all(
            string=True
        ):

            if isinstance(
                node,
                (Comment, Doctype),
            ):
                continue

            parent = node.parent

            if not parent:
                continue

            if parent.name in SKIP_TAGS:
                continue

            if node.strip():
                nodes.append(node)

    # --------------------------------------------------------
    # TOC
    # --------------------------------------------------------

    toc_entries = collect_toc_entries(book)

    # --------------------------------------------------------
    # Metadata title
    # --------------------------------------------------------

    metadata = book.get_metadata(
        "DC",
        "title",
    )

    title_text = (
        metadata[0][0]
        if metadata
        else None
    )

    # --------------------------------------------------------
    # Build translation list
    # --------------------------------------------------------

    texts = [
        str(node)
        for node in nodes
    ]

    texts += [
        str(entry.title)
        for entry in toc_entries
        if getattr(
            entry,
            "title",
            None,
        )
    ]

    if title_text:
        texts.append(title_text)

    log.info(
        "Documents=%d, text nodes=%d, total strings=%d",
        len(docs),
        len(nodes),
        len(texts),
    )

    # --------------------------------------------------------
    # Translate
    # --------------------------------------------------------

    translated = translate_many(
        texts,
        target,
        progress=progress,
    )

    # --------------------------------------------------------
    # Replace HTML text
    # --------------------------------------------------------

    pos = 0

    for node in nodes:

        node.replace_with(
            translated[pos]
        )

        pos += 1

    # --------------------------------------------------------
    # Replace TOC
    # --------------------------------------------------------

    for entry in toc_entries:

        if getattr(
            entry,
            "title",
            None,
        ):

            entry.title = translated[pos]

            pos += 1

    # --------------------------------------------------------
    # Replace title
    # --------------------------------------------------------

    if title_text:

        new_title = translated[pos]

        try:

            book.set_unique_metadata(
                "DC",
                "title",
                new_title,
            )

        except Exception:

            book.set_metadata(
                "DC",
                "title",
                new_title,
            )

    # --------------------------------------------------------
    # Language metadata
    # --------------------------------------------------------

    try:
        book.set_language(target)
    except Exception:
        pass

    # --------------------------------------------------------
    # Write HTML back
    # --------------------------------------------------------

    for item, soup in zip(
        docs,
        soups,
    ):

        item.set_content(
            str(soup).encode("utf-8")
        )

    # --------------------------------------------------------
    # Write EPUB
    # --------------------------------------------------------

    epub.write_epub(
        dst,
        book,
        {
            "raise_exceptions": True
        },
    )

    log.info(
        "EPUB created: %s",
        dst,
    )


# ============================================================
# FILE NAME
# ============================================================

def safe_filename(
    name,
    target,
):

    stem = Path(name).stem

    stem = re.sub(
        r'[\\/:*?"<>|]+',
        "_",
        stem,
    ).strip()

    if not stem:
        stem = "translated_book"

    return f"{stem}_{target}.epub"


# ============================================================
# /START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    context.user_data.clear()

    await update.message.reply_text(
        "📚 EPUB Translator\n\n"
        "Send an .epub file, then choose "
        "the target language.\n\n"
        f"Maximum file size: {MAX_MB} MB."
    )


# ============================================================
# /HELP
# ============================================================

async def help_cmd(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    await update.message.reply_text(
        "📚 EPUB Translator Help\n\n"
        "1. Send an EPUB file.\n"
        "2. Choose the target language.\n"
        "3. Wait for translation.\n"
        "4. The translated EPUB will be returned.\n\n"
        "The bot tries to preserve:\n"
        "• Images\n"
        "• CSS\n"
        "• Chapter structure\n"
        "• EPUB metadata\n"
        "• Formatting\n\n"
        "Translation uses Google Translate "
        "through deep-translator."
    )


# ============================================================
# RECEIVE EPUB
# ============================================================

async def on_file(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    doc = update.message.document

    name = doc.file_name or ""

    # --------------------------------------------------------
    # Extension
    # --------------------------------------------------------

    if not name.lower().endswith(".epub"):

        await update.message.reply_text(
            "❌ Please send an .epub file."
        )

        return

    # --------------------------------------------------------
    # Size
    # --------------------------------------------------------

    if (
        doc.file_size
        and doc.file_size
        > MAX_MB * 1024 * 1024
    ):

        await update.message.reply_text(
            f"❌ File too large.\n"
            f"Maximum: {MAX_MB} MB."
        )

        return

    # --------------------------------------------------------
    # Save Telegram file ID
    # --------------------------------------------------------

    context.user_data["file_id"] = doc.file_id
    context.user_data["name"] = name

    # --------------------------------------------------------
    # Language buttons
    # --------------------------------------------------------

    items = list(LANGS.items())

    rows = []

    for i in range(
        0,
        len(items),
        3,
    ):

        rows.append(
            [
                InlineKeyboardButton(
                    label,
                    callback_data=f"lang:{code}",
                )

                for code, label
                in items[i:i + 3]
            ]
        )

    rows.append(
        [
            InlineKeyboardButton(
                "❌ Cancel",
                callback_data="cancel",
            )
        ]
    )

    await update.message.reply_text(
        "🌐 Choose target language:",
        reply_markup=InlineKeyboardMarkup(
            rows
        ),
    )


# ============================================================
# LANGUAGE SELECTION
# ============================================================

async def on_lang(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    query = update.callback_query

    await query.answer()

    # --------------------------------------------------------
    # Cancel
    # --------------------------------------------------------

    if query.data == "cancel":

        context.user_data.clear()

        await query.edit_message_text(
            "❌ Cancelled."
        )

        return

    # --------------------------------------------------------
    # Validate callback
    # --------------------------------------------------------

    if not query.data.startswith(
        "lang:"
    ):
        return

    lang = query.data.split(
        ":",
        1,
    )[1]

    if lang not in LANGS:

        await query.edit_message_text(
            "❌ Invalid language."
        )

        return

    # --------------------------------------------------------
    # Get stored file
    # --------------------------------------------------------

    file_id = context.user_data.get(
        "file_id"
    )

    name = context.user_data.get(
        "name",
        "book.epub",
    )

    if not file_id:

        await query.edit_message_text(
            "❌ Session expired.\n"
            "Please send the EPUB again."
        )

        return

    await query.edit_message_text(
        f"⏳ Starting translation → "
        f"{LANGS[lang]}\n\n"
        "Large novels may take some time."
    )

    # --------------------------------------------------------
    # Event loop
    # --------------------------------------------------------

    loop = asyncio.get_running_loop()

    progress_message = query.message

    last_pct = -1
    last_update = 0.0

    # --------------------------------------------------------
    # Progress callback
    # --------------------------------------------------------

    def progress(
        done,
        total,
    ):

        nonlocal last_pct
        nonlocal last_update

        pct = int(
            done * 100 / max(
                total,
                1,
            )
        )

        now = time.monotonic()

        if (
            pct >= 100
            or pct - last_pct >= 10
            or now - last_update >= 8
        ):

            last_pct = pct
            last_update = now

            future = asyncio.run_coroutine_threadsafe(
                progress_message.edit_text(
                    f"⏳ Translating… {pct}%"
                ),
                loop,
            )

            # We intentionally don't wait here.
            # Translation worker must continue.
            _ = future

    # --------------------------------------------------------
    # Temporary working directory
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="epubbot_"
    ) as tmp:

        src = str(
            Path(tmp) / "input.epub"
        )

        dst = str(
            Path(tmp)
            / safe_filename(
                name,
                lang,
            )
        )

        try:

            # ------------------------------------------------
            # Download Telegram file
            # ------------------------------------------------

            tg_file = await context.bot.get_file(
                file_id
            )

            await tg_file.download_to_drive(
                src
            )

            # ------------------------------------------------
            # Validate EPUB container
            # ------------------------------------------------

            if not zipfile.is_zipfile(src):

                raise ValueError(
                    "Uploaded file is not a valid EPUB/ZIP container."
                )

            # ------------------------------------------------
            # Run translation outside async loop
            # ------------------------------------------------

            await loop.run_in_executor(
                None,
                translate_epub,
                src,
                dst,
                lang,
                progress,
            )

            # ------------------------------------------------
            # Validate output
            # ------------------------------------------------

            if not os.path.exists(dst):

                raise ValueError(
                    "Output EPUB was not created."
                )

            size = os.path.getsize(dst)

            if size == 0:

                raise ValueError(
                    "Output EPUB is empty."
                )

            # ------------------------------------------------
            # Upload
            # ------------------------------------------------

            await progress_message.edit_text(
                "📦 Translation finished.\n"
                "Uploading EPUB…"
            )

            with open(
                dst,
                "rb",
            ) as fh:

                await context.bot.send_document(
                    chat_id=query.message.chat_id,
                    document=fh,
                    filename=Path(dst).name,
                    caption=(
                        f"✅ Translation complete\n"
                        f"Language: {LANGS[lang]}"
                    ),
                )

            # ------------------------------------------------
            # Remove progress message
            # ------------------------------------------------

            try:

                await progress_message.delete()

            except Exception:

                pass

        except Exception:

            log.exception(
                "Translation failed"
            )

            await progress_message.edit_text(
                "❌ Translation failed.\n\n"
                "Your original EPUB was not modified.\n"
                "Please try again."
            )

        finally:

            context.user_data.clear()


# ============================================================
# RENDER HEALTH SERVER
# ============================================================

class HealthHandler(
    BaseHTTPRequestHandler
):

    def do_GET(self):

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/plain",
        )

        self.end_headers()

        self.wfile.write(
            b"ok"
        )

    def log_message(
        self,
        *args,
    ):
        pass


def start_health_server():

    port = os.getenv("PORT")

    if not port:
        return

    server = HTTPServer(
        (
            "0.0.0.0",
            int(port),
        ),
        HealthHandler,
    )

    threading.Thread(
        target=server.serve_forever,
        daemon=True,
    ).start()

    log.info(
        "Health server running on port %s",
        port,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_health_server()

    # Python 3.14 compatibility.
    #
    # python-telegram-bot 21.x can expect a current
    # event loop when run_polling() starts.
    #
    # Explicitly create one if none exists.

    try:

        asyncio.get_event_loop()

    except RuntimeError:

        asyncio.set_event_loop(
            asyncio.new_event_loop()
        )

    # --------------------------------------------------------
    # Telegram application
    # --------------------------------------------------------

    app = (
        Application
        .builder()
        .token(TOKEN)
        .concurrent_updates(True)
        .build()
    )

    # --------------------------------------------------------
    # Handlers
    # --------------------------------------------------------

    app.add_handler(
        CommandHandler(
            "start",
            start,
        )
    )

    app.add_handler(
        CommandHandler(
            "help",
            help_cmd,
        )
    )

    app.add_handler(
        MessageHandler(
            filters.Document.ALL,
            on_file,
        )
    )

    app.add_handler(
        CallbackQueryHandler(
            on_lang
        )
    )

    log.info(
        "EPUB translator bot started"
    )

    # --------------------------------------------------------
    # Start polling
    # --------------------------------------------------------

    app.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()
