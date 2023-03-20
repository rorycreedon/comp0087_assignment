def chunk_text(text, max_length=512):
    """
    Split a large piece of text into chunks that are no longer than max_length words,
    but do not cut off sentences.
    """
    # Split the text into sentences
    sentences = text.split('. ')

    # Initialize variables
    chunks = []
    current_chunk = ''

    # Loop through each sentence and add it to the current chunk until the
    # length of the current chunk exceeds the max_length
    for sentence in sentences:
        words = sentence.split()
        if len(current_chunk.split()) + len(words) <= max_length:
            current_chunk += sentence + '. '
        else:
            # Add the current chunk to the list of chunks
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '

    # Add the final chunk to the list of chunks
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks