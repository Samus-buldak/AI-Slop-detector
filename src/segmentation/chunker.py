# src/segmentation/chunker.py
from typing import List

def chunk_text(text: str, max_chars: int = 2500, overlap: int = 300) -> List[str]:
    """
    Разбивает текст на чанки с перекрытием, стараясь не резать слова.
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + max_chars
        
        # Если это не последний кусок, ищем границу слова или предложения
        if end < text_len:
            # Ищем ближайший пробел или перенос строки перед концом чанка
            split_point = text.rfind(' ', start, end)
            if split_point == -1 or split_point < start + (max_chars // 2):
                split_point = end # Если не нашли, режем жестко
            else:
                split_point += 1 # Включаем пробел в предыдущий чанк или исключаем? Лучше исключить из следующего
                end = split_point
                
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        # Сдвигаем начало следующего чанка с учетом overlap
        start = end - overlap
        if start <= 0: break # Защита от бесконечного цикла
        
    return chunks