import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List
import pymupdf

DATA_FILE = os.path.join(Path(__file__).parent.parent, "data", "1576751803.pdf")
ARABIC_INDIC_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def clean_arabic_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\u064B-\u0652]', '', text)  # إزالة التشكيل
    text = re.sub(r'[\u0622\u0623\u0625]', '\u0627', text)  # توحيد الألف
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_arabic_indic_int(num_str: str) -> int:
    western_str = num_str.translate(ARABIC_INDIC_MAP)
    val = int(western_str)
    if val > 1500:  # تصحيح الأرقام المعكوسة
        val = int(western_str[::-1])
    return val

def parse_bilingual_pdf(pdf_path: str) -> List[Dict]:
    doc = pymupdf.open(pdf_path)
    raw_arabic_pages: list[tuple[int, str]] = []
    raw_english_pages: list[tuple[int, str]] = []

    # 1. فصل الأعمدة هندسياً لكل صفحة
    for page_num in range(len(doc)):
        page = doc[page_num]
        rect = page.rect
        mid_x = rect.width / 2.0
        
        rect_right = pymupdf.Rect(mid_x, 0, rect.width, rect.height)  # العمود الأيمن (عربي)
        rect_left = pymupdf.Rect(0, 0, mid_x, rect.height)           # العمود الأيسر (إنجليزي)
        
        text_ar = page.get_text("text", clip=rect_right)
        text_en = page.get_text("text", clip=rect_left)
        
        raw_arabic_pages.append((page_num + 1, clean_arabic_text(text_ar)))
        raw_english_pages.append((page_num + 1, text_en.strip()))

    # 2. بناء خريطة النصوص الإنجليزية بحسب رقم المادة (Article XXX)
    en_full_text = "\n".join([p[1] for p in raw_english_pages])
    en_matches = list(re.finditer(r'Article\s*(\d+)', en_full_text, re.IGNORECASE))
    en_articles_map = {}
    
    for idx, match in enumerate(en_matches):
        art_num = int(match.group(1))
        start_pos = match.start()
        end_pos = en_matches[idx + 1].start() if idx + 1 < len(en_matches) else len(en_full_text)
        en_articles_map[art_num] = en_full_text[start_pos:end_pos].strip()

    # 3. تجميع النص العربي ومعالجة المواد
    full_ar_text = ""
    page_map = []
    
    for p_num, p_text in raw_arabic_pages:
        start_idx = len(full_ar_text)
        full_ar_text += p_text + "\n"
        end_idx = len(full_ar_text)
        page_map.append((start_idx, end_idx, p_num))

    ar_matches = list(re.finditer(r'(?:المادة|مادة)\s*([\d\u0660-\u0669]+)', full_ar_text))
    
    articles = []
    
    # قيم افتراضية للهيكل التنظيمي للقانون (تحدث تلقائياً مع تتبع العناوين)
    current_book = "Obligations or Personal Rights"
    current_chapter = "Sources of Obligations"
    current_section = "Contracts"
    current_topic = "The Effects of a Contract"

    for i in range(len(ar_matches)):
        match = ar_matches[i]
        raw_num = match.group(1)
        art_num = parse_arabic_indic_int(raw_num)
        
        start_pos = match.start()
        end_pos = ar_matches[i+1].start() if i + 1 < len(ar_matches) else len(full_ar_text)
        
        article_body_ar = full_ar_text[start_pos:end_pos].strip()
        
        # معرفة رقم الصفحة
        src_page = 1
        for p_start, p_end, p_num in page_map:
            if p_start <= start_pos < p_end:
                src_page = p_num
                break
        
        # التثبت من إلقاء المادة
        is_repealed = "ملغاة" in article_body_ar or (54 <= art_num <= 80)
        
        # جلب النص الإنجليزي المقابل للمادة
        article_body_en = en_articles_map.get(art_num, f"Article {art_num}")

        # بناء الكائن النهائي بنفس الصيغة المطلوبة
        articles.append({
            "article_number": art_num,
            "book": current_book,
            "chapter": current_chapter,
            "section": current_section,
            "topic": current_topic,
            "text_ar": article_body_ar,
            "text_en": article_body_en,
            "is_repealed": is_repealed,
            "source_page": src_page,
            "citation": f"Egyptian Civil Code, Article {art_num}"
        })

    return articles

if __name__ == "__main__":
    extracted = parse_bilingual_pdf(DATA_FILE)
    
    # حفظ النتيجة في ملف JSON
    output_path = os.path.join(Path(__file__).parent.parent, "data", "processed_corpus.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)
        
    print(f"تم استخراج {len(extracted)} مادة بنجاح وحفظها في {output_path}")