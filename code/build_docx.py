"""Build the manuscript .docx from a light markup file.
Markup: '# ' H1, '## ' H2, '### ' H3, 'TABLE|<csv>|<caption>|<note>' , 'FIG|<png>|<caption>|<width_in>',
'PAGEBREAK', '- ' bullet, blank line separates paragraphs. Inline *italic* and **bold**.
usage: build_docx.py MARKUP OUT.docx"""
import sys, re, csv
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Twips
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

src, out = sys.argv[1:3]
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for side in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'): setattr(sec, side, Inches(1))
st = doc.styles['Normal']; st.font.name = 'Times New Roman'; st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE; st.paragraph_format.space_after = Pt(0)
for lvl, size in ((1, 13), (2, 12), (3, 12)):
    h = doc.styles[f'Heading {lvl}']; h.font.name = 'Times New Roman'; h.font.size = Pt(size); h.font.bold = True
    h.font.italic = (lvl == 3); h.font.color.rgb = RGBColor(0, 0, 0)
    h.element.rPr.rFonts.set(qn('w:asciiTheme'), 'Times New Roman') if False else None
    rf = h.element.rPr.rFonts; [rf.attrib.pop(qn(a), None) for a in ('w:asciiTheme', 'w:hAnsiTheme', 'w:eastAsiaTheme', 'w:cstheme')]
    rf.set(qn('w:ascii'), 'Times New Roman'); rf.set(qn('w:hAnsi'), 'Times New Roman')
    h.paragraph_format.space_before = Pt(12); h.paragraph_format.space_after = Pt(6); h.paragraph_format.keep_with_next = True
    h.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE

# line numbers (requested by most Elsevier journals at review stage) and page numbers
ln = OxmlElement('w:lnNumType'); ln.set(qn('w:countBy'), '1'); ln.set(qn('w:restart'), 'continuous'); sec._sectPr.append(ln)
fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
for t, v in (('begin', None), (None, 'PAGE'), ('end', None)):
    r = fp.add_run()
    if t: e = OxmlElement('w:fldChar'); e.set(qn('w:fldCharType'), t); r._r.append(e)
    else: e = OxmlElement('w:instrText'); e.set(qn('xml:space'), 'preserve'); e.text = v; r._r.append(e)

def add_runs(p, text, size=None, base_bold=False):
    for part in re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|~[^~]+~)', text):
        if not part: continue
        if part.startswith('**'): r = p.add_run(part[2:-2]); r.bold = True
        elif part.startswith('*'): r = p.add_run(part[1:-1]); r.italic = True
        elif part.startswith('^'): r = p.add_run(part[1:-1]); r.font.superscript = True
        elif part.startswith('~'): r = p.add_run(part[1:-1]); r.font.subscript = True
        else: r = p.add_run(part)
        if base_bold: r.bold = True
        if size: r.font.size = Pt(size)

def shade(cell, hexcol):
    tcPr = cell._tc.get_or_add_tcPr(); s = OxmlElement('w:shd'); s.set(qn('w:val'), 'clear'); s.set(qn('w:color'), 'auto'); s.set(qn('w:fill'), hexcol); tcPr.append(s)

def borders(table):
    tbl = table._tbl; tblPr = tbl.tblPr; b = OxmlElement('w:tblBorders')
    for edge in ('top', 'bottom'):
        e = OxmlElement(f'w:{edge}'); e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '8'); e.set(qn('w:color'), '000000'); b.append(e)
    for edge in ('left', 'right', 'insideV', 'insideH'):
        e = OxmlElement(f'w:{edge}'); e.set(qn('w:val'), 'nil'); b.append(e)
    tblPr.append(b)

def add_table(path, caption, note):
    p = doc.add_paragraph(); add_runs(p, caption, size=11); p.paragraph_format.keep_with_next = True
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE; p.paragraph_format.space_before = Pt(12); p.paragraph_format.space_after = Pt(4)
    rows = list(csv.reader(open(path, encoding='utf-8')))
    t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.alignment = WD_TABLE_ALIGNMENT.CENTER; borders(t)
    tw = OxmlElement('w:tblW'); tw.set(qn('w:w'), '5000'); tw.set(qn('w:type'), 'pct')
    old = t._tbl.tblPr.find(qn('w:tblW'))
    if old is not None: t._tbl.tblPr.remove(old)
    t._tbl.tblPr.append(tw)
    ncol = len(rows[0]); total = 9360
    first = int(total * (0.22 if ncol > 4 else 0.3)); rest = (total - first) // (ncol - 1)
    for i, row in enumerate(t.rows):
        for j, c in enumerate(row.cells): c.width = Twips(first if j == 0 else rest)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.text = ''; q = c.paragraphs[0]
            q.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            q.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            add_runs(q, val, size=9, base_bold=(i == 0))
            if i == 0:
                tcPr = c._tc.get_or_add_tcPr(); bb = OxmlElement('w:tcBorders'); e = OxmlElement('w:bottom')
                e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '6'); e.set(qn('w:color'), '000000'); bb.append(e); tcPr.append(bb)
    if note:
        p = doc.add_paragraph(); add_runs(p, note, size=9); p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        p.paragraph_format.space_after = Pt(12)

def add_fig(path, caption, width):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Inches(float(width)))
    p = doc.add_paragraph(); add_runs(p, caption, size=11); p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.paragraph_format.space_after = Pt(12)

blocks = open(src, encoding='utf-8').read().split('\n')
para = []
def flush():
    global para
    if para:
        text = ' '.join(x.strip() for x in para)
        p = doc.add_paragraph(); add_runs(p, text)
        p.paragraph_format.first_line_indent = Inches(0.3) if not text.startswith(('Keywords', 'Highlights', '*Corresponding')) else None
    para = []
for line in blocks:
    s = line.rstrip()
    if s.startswith('<!--'): continue
    if not s.strip(): flush(); continue
    if s.startswith('TITLE|'):
        flush(); p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; add_runs(p, s[6:], size=14, base_bold=True); continue
    if s.startswith('CENTER|'):
        flush(); p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; add_runs(p, s[7:]); continue
    if s.startswith('### '): flush(); doc.add_heading(s[4:], 3); continue
    if s.startswith('## '): flush(); doc.add_heading(s[3:], 2); continue
    if s.startswith('# '): flush(); doc.add_heading(s[2:], 1); continue
    if s.startswith('TABLE|'): flush(); _, a, b, *c = s.split('|'); add_table(a, b, c[0] if c else ''); continue
    if s.startswith('FIG|'): flush(); _, a, b, c = s.split('|'); add_fig(a, b, c); continue
    if s == 'PAGEBREAK': flush(); doc.add_page_break(); continue
    if s.startswith('- '):
        flush(); p = doc.add_paragraph(style='List Bullet'); add_runs(p, s[2:]); continue
    if s.startswith('REF|'):
        flush(); p = doc.add_paragraph(); add_runs(p, s[4:]); p.paragraph_format.left_indent = Inches(0.3); p.paragraph_format.first_line_indent = Inches(-0.3); continue
    para.append(s)
flush()
import datetime
cp = doc.core_properties
cp.author = 'Nikhil Kottapalli'; cp.last_modified_by = 'Nikhil Kottapalli'; cp.comments = ''; cp.keywords = ''
cp.subject = ''; cp.category = ''; _first = open(src, encoding='utf-8').read().split('\n')[0]; cp.title = _first.split('|', 1)[1][:250] if _first.startswith('TITLE|') else ''
now = datetime.datetime.now(); cp.created = now; cp.modified = now; cp.revision = 1
doc.save(out); print('saved', out)
