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

# More workers = faster, but too many can trigger rate limits.
WORKERS = max(
    1,
    min(
        int(os.getenv("TRANSLATE_WORKERS", "6")),
        10,
    ),
)

CHUNK = max(
    1000,
    min(
        int(os.getenv("CHUNK_SIZE", "4200")),
        4600,
    ),
)

BATCH_CHARS = max(
    1800,
    min(
        int(os.getenv("BATCH_CHARS", "4200")),
        4600,
    ),
)

RETRIES = max(
    1,
    min(
        int(os.getenv("TRANSLATE_RETRIES", "4")),
        7,
    ),
)

UPDATE_SECONDS = max(
    2,
    float(
        os.getenv(
            "PROGRESS_UPDATE_SECONDS",
            "3",
        )
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


# ============================================================
# HTML
# ============================================================

SKIP_TAGS = {
    "script",
    "style",
    "code",
    "pre",
    "noscript",
    "svg",
}

_ws = re.compile(r"[ \t]+")
WORD_RE = re.compile(r"\S+")

MARKER_RE = re.compile(
    r"\[\[EPUBSEG(\d{1,7})\]\]"
)


# ============================================================
# WORD COUNT
# ============================================================

def word_count(text: str) -> int:
    return len(
        WORD_RE.findall(text)
    )


# ============================================================
# SPLIT LONG TEXT
# ============================================================

def split_long_text(
    text: str,
    size: int = CHUNK,
):

    text = text.strip()

    while len(text) > size:

        cut = text.rfind(
            " ",
            0,
            size,
        )

        if cut < int(size * 0.55):
            cut = size

        yield text[:cut]

        text = text[cut:].lstrip()

    if text:
        yield text


# ============================================================
# WHITESPACE
# ============================================================

def restore_ws(
    original: str,
    translated: str,
) -> str:

    lead = original[
        : len(original)
        - len(original.lstrip())
    ]

    trail = original[
        len(original.rstrip()):
    ]

    return (
        lead
        + translated
        + trail
    )


# ============================================================
# SINGLE TRANSLATION FALLBACK
# ============================================================

def translate_one(
    translator,
    text: str,
) -> str:

    if not text.strip():
        return text

    parts = []

    for part in split_long_text(text):

        result = None

        for attempt in range(RETRIES):

            try:

                result = translator.translate(
                    part
                )

                if result:
                    break

            except Exception as exc:

                delay = min(
                    8.0,
                    0.8 * (2 ** attempt),
                )

                log.warning(
                    "translation retry %d/%d: %s",
                    attempt + 1,
                    RETRIES,
                    exc,
                )

                time.sleep(delay)

        # Never lose original text.
        parts.append(
            result if result else part
        )

    return " ".join(parts)


# ============================================================
# BATCH TRANSLATION
# ============================================================

def translate_marked_batch(
    items,
    target,
):
    """
    Translate many text segments using one request.

    Markers let us map translated text back
    to the original EPUB text nodes.
    """

    translator = GoogleTranslator(
        source="auto",
        target=target,
    )

    payload = "\n".join(
        f"[[EPUBSEG{i}]] {text}"
        for i, text in items
    )

    for attempt in range(RETRIES):

        try:

            result = translator.translate(
                payload
            )

            if result:

                matches = list(
                    MARKER_RE.finditer(
                        result
                    )
                )

                if (
                    matches
                    and len(matches)
                    == len(items)
                ):

                    out = {}

                    for pos, match in enumerate(
                        matches
                    ):

                        start = match.end()

                        if (
                            pos + 1
                            < len(matches)
                        ):
                            end = matches[
                                pos + 1
                            ].start()

                        else:
                            end = len(
                                result
                            )

                        translated = (
                            result[
                                start:end
                            ].strip()
                        )

                        out[
                            items[pos][0]
                        ] = translated

                    if all(
                        k in out
                        and out[k]
                        for k, _ in items
                    ):
                        return out

        except Exception as exc:

            delay = min(
                8.0,
                0.8 * (2 ** attempt),
            )

            log.warning(
                "batch retry %d/%d: %s",
                attempt + 1,
                RETRIES,
                exc,
            )

            time.sleep(delay)

    # --------------------------------------------------------
    # Safe fallback
    # --------------------------------------------------------

    log.warning(
        "Batch mapping failed; "
        "falling back to individual translation."
    )

    return {
        idx: translate_one(
            translator,
            text,
        )
        for idx, text in items
    }


# ============================================================
# BUILD BATCHES
# ============================================================

def build_batches(texts):

    batches = []

    current = []

    chars = 0

    seg_id = 0

    for original_index, raw in enumerate(
        texts
    ):

        if not raw.strip():
            continue

        normalized = _ws.sub(
            " ",
            raw.strip(),
        )

        pieces = list(
            split_long_text(
                normalized,
                CHUNK,
            )
        )

        for piece in pieces:

            # Marker overhead.
            cost = len(piece) + 18

            if (
                current
                and chars + cost + 1
                > BATCH_CHARS
            ):

                batches.append(
                    current
                )

                current = []

                chars = 0

            current.append(
                (
                    seg_id,
                    piece,
                    original_index,
                )
            )

            seg_id += 1

            chars += (
                cost + 1
            )

    if current:
        batches.append(
            current
        )

    return batches


# ============================================================
# TRANSLATE MANY
# ============================================================

def translate_many(
    texts,
    target,
    progress=None,
):

    out = list(texts)

    batches = build_batches(
        texts
    )

    total_words = sum(
        word_count(t)
        for t in texts
    )

    if not batches:

        if progress:

            progress(
                0,
                0,
                total_words,
                total_words,
                0,
            )

        return (
            out,
            total_words,
        )

    completed_words = 0

    completed_batches = 0

    total_batches = len(
        batches
    )

    started = time.monotonic()

    lock = threading.Lock()

    def worker(batch):

        request_items = [
            (
                seg_id,
                text,
            )

            for (
                seg_id,
                text,
                _,
            ) in batch
        ]

        translated_map = (
            translate_marked_batch(
                request_items,
                target,
            )
        )

        return (
            batch,
            translated_map,
        )

    # --------------------------------------------------------
    # Parallel batches
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        futures = [
            executor.submit(
                worker,
                batch,
            )

            for batch in batches
        ]

        for future in as_completed(
            futures
        ):

            batch, translated_map = (
                future.result()
            )

            batch_words = 0

            for (
                seg_id,
                original_text,
                original_index,
            ) in batch:

                translated = (
                    translated_map.get(
                        seg_id,
                        original_text,
                    )
                )

                # A node can contain multiple chunks.
                if (
                    out[original_index]
                    == texts[original_index]
                ):

                    out[
                        original_index
                    ] = translated

                else:

                    out[
                        original_index
                    ] += (
                        " "
                        + translated
                    )

                batch_words += word_count(
                    original_text
                )

            with lock:

                completed_batches += 1

                completed_words += (
                    batch_words
                )

                elapsed = max(
                    0.001,
                    time.monotonic()
                    - started,
                )

                rate = (
                    completed_words
                    / elapsed
                )

                remaining = max(
                    0,
                    total_words
                    - completed_words,
                )

                eta = (
                    remaining / rate
                    if rate > 0
                    else 0
                )

                if progress:

                    progress(
                        completed_batches,
                        total_batches,
                        total_words,
                        completed_words,
                        eta,
                    )

    # --------------------------------------------------------
    # Restore whitespace
    # --------------------------------------------------------

    for i, original in enumerate(
        texts
    ):

        if original.strip():

            out[i] = restore_ws(
                original,
                out[i],
            )

    return (
        out,
        total_words,
    )


# ============================================================
# EPUB TOC
# ============================================================

def collect_toc_entries(book):

    entries = []

    def walk(items):

        for item in items:

            if isinstance(
                item,
                tuple,
            ):

                if item:
                    entries.append(
                        item[0]
                    )

                if len(item) > 1:
                    walk(
                        item[1]
                    )

            elif hasattr(
                item,
                "title",
            ):

                entries.append(
                    item
                )

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

    book = epub.read_epub(
        src
    )

    docs = [
        item

        for item
        in book.get_items()

        if item.get_type()
        == ITEM_DOCUMENT
    ]

    soups = []

    nodes = []

    # --------------------------------------------------------
    # Parse HTML
    # --------------------------------------------------------

    for item in docs:

        soup = BeautifulSoup(
            item.get_content(),
            "html.parser",
        )

        soups.append(
            soup
        )

        for node in soup.find_all(
            string=True
        ):

            if isinstance(
                node,
                (
                    Comment,
                    Doctype,
                ),
            ):
                continue

            parent = node.parent

            if not parent:
                continue

            if (
                parent.name
                in SKIP_TAGS
            ):
                continue

            if node.strip():

                nodes.append(
                    node
                )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    toc_entries = (
        collect_toc_entries(book)
    )

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
    # Body text
    # --------------------------------------------------------

    body_texts = [
        str(node)
        for node in nodes
    ]

    translated_body, body_words = (
        translate_many(
            body_texts,
            target,
            progress=progress,
        )
    )

    # --------------------------------------------------------
    # Replace body text
    # --------------------------------------------------------

    for (
        node,
        translated,
    ) in zip(
        nodes,
        translated_body,
    ):

        node.replace_with(
            translated
        )

    # --------------------------------------------------------
    # TOC + TITLE
    # --------------------------------------------------------

    extras = []

    for entry in toc_entries:

        if getattr(
            entry,
            "title",
            None,
        ):

            extras.append(
                str(entry.title)
            )

    if title_text:

        extras.append(
            title_text
        )

    if extras:

        extra_translated, _ = (
            translate_many(
                extras,
                target,
                progress=None,
            )
        )

        p = 0

        for entry in toc_entries:

            if getattr(
                entry,
                "title",
                None,
            ):

                entry.title = (
                    extra_translated[p]
                )

                p += 1

        if title_text:

            try:

                book.set_unique_metadata(
                    "DC",
                    "title",
                    extra_translated[p],
                )

            except Exception:

                book.set_metadata(
                    "DC",
                    "title",
                    extra_translated[p],
                )

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    try:

        book.set_language(
            target
        )

    except Exception:

        pass

    # --------------------------------------------------------
    # Save HTML
    # --------------------------------------------------------

    for (
        item,
        soup,
    ) in zip(
        docs,
        soups,
    ):

        item.set_content(
            str(soup).encode(
                "utf-8"
            )
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

    return body_words


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

        stem = (
            "translated_book"
        )

    return (
        f"{stem}_{target}.epub"
    )


# ============================================================
# TIME FORMAT
# ============================================================

def format_seconds(
    seconds,
):

    seconds = max(
        0,
        int(seconds),
    )

    h, rem = divmod(
        seconds,
        3600,
    )

    m, s = divmod(
        rem,
        60,
    )

    if h:

        return (
            f"{h}h "
            f"{m:02d}m "
            f"{s:02d}s"
        )

    if m:

        return (
            f"{m}m "
            f"{s:02d}s"
        )

    return f"{s}s"


# ============================================================
# PROGRESS BAR
# ============================================================

def progress_bar(
    percent,
    width=20,
):

    filled = int(
        width
        * percent
        / 100
    )

    return (
        "█" * filled
        + "░" * (
            width - filled
        )
    )


# ============================================================
# PROGRESS MESSAGE
# ============================================================

def progress_text(
    total_words,
    done_words,
    started,
    eta,
):

    elapsed = max(
        0.0,
        time.monotonic()
        - started,
    )

    pct = min(
        100,
        int(
            done_words
            * 100
            / max(
                1,
                total_words,
            )
        ),
    )

    remaining = max(
        0,
        total_words
        - done_words,
    )

    rate = (
        done_words / elapsed
        if elapsed > 0
        else 0
    )

    return (
        "📚 **Translating EPUB**\n\n"

        f"`{progress_bar(pct)}` "
        f"**{pct}%**\n\n"

        f"⏱ **Estimated time:** "
        f"{format_seconds(eta)}\n"

        f"⏳ **Used time:** "
        f"{format_seconds(elapsed)}\n\n"

        f"📄 **File Words:** "
        f"{total_words:,}\n"

        f"✅ **Translated words:** "
        f"{done_words:,}\n"

        f"📝 **Remaining words:** "
        f"{remaining:,}\n\n"

        f"⚡ **Speed:** "
        f"{rate:,.1f} words/sec"
    )


# ============================================================
# START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    context.user_data.clear()

    await update.message.reply_text(
        "📚 EPUB Translator\n\n"
        "Send an .epub file and "
        "choose the target language.\n\n"
        f"Maximum file size: "
        f"{MAX_MB} MB."
    )


# ============================================================
# HELP
# ============================================================

async def help_cmd(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    await update.message.reply_text(
        "📚 EPUB Translator Help\n\n"

        "1. Send an EPUB.\n"
        "2. Select the language.\n"
        "3. Watch live progress.\n"
        "4. Receive the translated EPUB.\n\n"

        "⚡ Fast engine:\n"
        "• Batch translation\n"
        "• Parallel workers\n"
        "• Automatic retries\n"
        "• ETA\n"
        "• Words/sec\n"
        "• Remaining words"
    )


# ============================================================
# EPUB FILE
# ============================================================

async def on_file(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    doc = update.message.document

    name = (
        doc.file_name
        or ""
    )

    if not name.lower().endswith(
        ".epub"
    ):

        await update.message.reply_text(
            "❌ Please send an .epub file."
        )

        return

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

    context.user_data[
        "file_id"
    ] = doc.file_id

    context.user_data[
        "name"
    ] = name

    items = list(
        LANGS.items()
    )

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
                    callback_data=(
                        f"lang:{code}"
                    ),
                )

                for code, label
                in items[
                    i:i + 3
                ]
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
        reply_markup=(
            InlineKeyboardMarkup(
                rows
            )
        ),
    )


# ============================================================
# LANGUAGE CALLBACK
# ============================================================

async def on_lang(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    query = update.callback_query

    await query.answer()

    if query.data == "cancel":

        context.user_data.clear()

        await query.edit_message_text(
            "❌ Cancelled."
        )

        return

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

    file_id = (
        context.user_data.get(
            "file_id"
        )
    )

    name = (
        context.user_data.get(
            "name",
            "book.epub",
        )
    )

    if not file_id:

        await query.edit_message_text(
            "❌ Session expired.\n"
            "Send the EPUB again."
        )

        return

    await query.edit_message_text(
        f"⏳ Preparing translation → "
        f"{LANGS[lang]}\n\n"
        "Analyzing EPUB…"
    )

    loop = (
        asyncio.get_running_loop()
    )

    message = query.message

    started = time.monotonic()

    state = {
        "last": 0.0,
        "last_text": "",
    }

    total_words_holder = {
        "value": 0
    }

    # --------------------------------------------------------
    # Telegram UI
    # --------------------------------------------------------

    async def update_ui(
        text,
    ):

        try:

            await message.edit_text(
                text,
                parse_mode="Markdown",
            )

        except Exception as exc:

            log.debug(
                "progress update skipped: %s",
                exc,
            )

    # --------------------------------------------------------
    # Progress callback
    # --------------------------------------------------------

    def progress(
        done_batches,
        total_batches,
        total_words,
        done_words,
        eta,
    ):

        total_words_holder[
            "value"
        ] = total_words

        now = time.monotonic()

        # Don't hammer Telegram.
        if (
            done_words < total_words
            and
            now - state["last"]
            < UPDATE_SECONDS
        ):

            return

        state["last"] = now

        text = progress_text(
            total_words,
            done_words,
            started,
            eta,
        )

        if (
            text
            == state["last_text"]
        ):

            return

        state["last_text"] = text

        asyncio.run_coroutine_threadsafe(
            update_ui(text),
            loop,
        )

    # --------------------------------------------------------
    # Temporary directory
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="epubbot_"
    ) as tmp:

        src = str(
            Path(tmp)
            / "input.epub"
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
            # Download
            # ------------------------------------------------

            tg_file = (
                await context.bot.get_file(
                    file_id
                )
            )

            await tg_file.download_to_drive(
                src
            )

            # ------------------------------------------------
            # Validate
            # ------------------------------------------------

            if not zipfile.is_zipfile(
                src
            ):

                raise ValueError(
                    "Uploaded file is not "
                    "a valid EPUB/ZIP container."
                )

            # ------------------------------------------------
            # Translation
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

            if (
                not os.path.exists(dst)
                or os.path.getsize(dst)
                == 0
            ):

                raise ValueError(
                    "Output EPUB was not "
                    "created correctly."
                )

            # ------------------------------------------------
            # Final progress
            # ------------------------------------------------

            final_text = progress_text(
                total_words_holder[
                    "value"
                ],
                total_words_holder[
                    "value"
                ],
                started,
                0,
            )

            final_text = (
                final_text.replace(
                    "**0%**",
                    "**100%**",
                    1,
                )
            )

            final_text += (
                "\n\n"
                "📦 Uploading translated EPUB…"
            )

            await update_ui(
                final_text
            )

            # ------------------------------------------------
            # Send EPUB
            # ------------------------------------------------

            with open(
                dst,
                "rb",
            ) as fh:

                await context.bot.send_document(
                    chat_id=(
                        query.message.chat_id
                    ),
                    document=fh,
                    filename=(
                        Path(dst).name
                    ),
                    caption=(
                        "✅ Translation complete — "
                        f"{LANGS[lang]}"
                    ),
                )

            try:

                await message.delete()

            except Exception:

                pass

        except Exception:

            log.exception(
                "Translation failed"
            )

            await update_ui(
                "❌ Translation failed.\n\n"
                "The original EPUB was not modified.\n"
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

        self.send_response(
            200
        )

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

    port = os.getenv(
        "PORT"
    )

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
    try:

        asyncio.get_event_loop()

    except RuntimeError:

        asyncio.set_event_loop(
            asyncio.new_event_loop()
        )

    app = (
        Application.builder()
        .token(TOKEN)
        .concurrent_updates(True)
        .build()
    )

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
        "EPUB translator bot started | "
        "workers=%d | batch_chars=%d | chunk=%d",
        WORKERS,
        BATCH_CHARS,
        CHUNK,
    )

    app.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()
