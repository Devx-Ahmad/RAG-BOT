from docx import Document

def load_docx_chunks(path):
    doc = Document(path)
    chunks = []

    # Dictionary to track current hierarchy
    heading_levels = {}

    # Mapping styles to levels
    allowed_levels = {
        "Heading 2": 2,
        "Heading 3": 3,
        "Heading 4": 4,
        "Heading 8": 8,
        "Heading 9": 9,
    }

    current_chunk_text = []

    def flush_chunk():
        if current_chunk_text:
            section_path = " > ".join(
                heading_levels[k] for k in sorted(heading_levels) if k in heading_levels
            )
            full_text = "\n".join(current_chunk_text).strip()
            if full_text:
                chunks.append(f"[{section_path}]\n{full_text}")
            current_chunk_text.clear()

    for para in doc.paragraphs:
        text = para.text.strip()
        style_name = para.style.name

        if not text:
            continue

        if style_name in allowed_levels:
            flush_chunk()  # Save current chunk before switching section
            level = allowed_levels[style_name]
            heading_levels[level] = text
            # Remove deeper levels
            keys_to_delete = [k for k in heading_levels if k > level]
            for k in keys_to_delete:
                del heading_levels[k]
        else:
            current_chunk_text.append(text)

    flush_chunk()  # flush last chunk

    return chunks
