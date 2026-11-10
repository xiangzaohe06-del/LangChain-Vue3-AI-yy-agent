"""Prepare a reviewed source set in a temporary import directory, without activating it.

Run from dietitian: python tools/expand_knowledge.py
Acquisition uses Windows system trust roots; no TLS verification is disabled.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from html.parser import HTMLParser
from io import BytesIO
import json
import logging
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pip._vendor.truststore as truststore
truststore.inject_into_ssl()
import requests
from pypdf import PdfReader
from rag.store import knowledge_store

TODAY = '2026-11-06'
NIH_POLICY = 'https://www.nih.gov/about-nih/frequently-asked-questions'
FDA_POLICY = 'https://www.fda.gov/about-fda/about-website/website-policies'
NIDDK = 'https://www.niddk.nih.gov/health-information/'
FDA = 'https://www.fda.gov/food/'
SOURCES = [
    ('weight_plan', '成人体重管理：健康饮食与身体活动', NIDDK+'weight-management/adult-overweight-obesity/eating-physical-activity', '成年人一般科普；能量目标需个体化'),
    ('adult_nutrition', '成人日常膳食：全谷蔬果蛋白质与活动习惯', NIDDK+'weight-management/healthy-eating-physical-activity-for-life/health-tips-for-adults', '成年人一般膳食教育'),
    ('portion_sizes', '份量管理：食物份量、餐盘和外出就餐', NIDDK+'weight-management/just-enough-food-portions', '一般人群份量认知；标签份量不等于个人建议摄入量'),
    ('habit_change', '饮食行为改变：制定目标、记录与长期维持', NIDDK+'weight-management/changing-habits-better-health', '生活方式教育；不代替疾病治疗'),
    ('child_weight', '儿童家庭饮食：成长、健康体重与家长支持', NIDDK+'weight-management/helping-your-child-who-is-overweight', '儿童需结合生长发育评估，不套用成人减重或热量限制'),
    ('teen_health', '青少年饮食与成长：早餐、活动和健康习惯', NIDDK+'weight-management/take-charge-your-health-guide-teenagers', '青少年科普，不适用成人限能量方案'),
    ('pregnancy', '孕期膳食与活动：营养需求和体重管理', NIDDK+'health-information/weight-management/health-tips-pregnant-women', '孕期需要产科个体化评估；不可自行极端减重'),
    ('lactose', '乳糖不耐受饮食：奶类、酸奶、乳糖与钙', NIDDK+'digestive-diseases/lactose-intolerance/eating-diet-nutrition', '乳糖不耐受科普，不等同牛奶蛋白过敏'),
    ('constipation', '便秘饮食：膳食纤维、水分和渐进调整', NIDDK+'digestive-diseases/constipation/eating-diet-nutrition', '相关症状需评估原因，不以饮食代替诊治'),
    ('ibs', '肠易激综合征饮食：纤维与低FODMAP适用范围', NIDDK+'digestive-diseases/irritable-bowel-syndrome/eating-diet-nutrition', '低FODMAP不是所有人通用的永久饮食；需专业指导'),
    ('celiac', '乳糜泻饮食：无麸质、标签和交叉接触', NIDDK+'digestive-diseases/celiac-disease/eating-diet-nutrition', '针对乳糜泻；不建议普通人一律无麸质'),
    ('reflux', '胃食管反流饮食：食物诱因与进餐时间', NIDDK+'digestive-diseases/acid-reflux-ger-gerd-adults/eating-diet-nutrition', '食物诱因存在个体差异，不能代替诊断'),
    ('diverticular', '憩室病饮食：膳食纤维与食物选择', NIDDK+'digestive-diseases/diverticular-disease/eating-diet-nutrition', '憩室病不同阶段需个体化，不自行处理急性腹痛'),
    ('diabetes', '糖尿病饮食：餐食计划、碳水化合物和身体活动', NIDDK+'diabetes/overview/diet-eating-physical-activity', '糖尿病科普；药物、胰岛素与碳水目标需医护指导'),
    ('labels', '营养标签：份量、能量、每日参考值与营养素比较', FDA+'nutrition-facts-label/how-understand-and-use-nutrition-facts-label', '美国FDA标签体系；不直接等同中国营养标签法规或DRIs'),
    ('sodium', '钠与食盐：隐形钠、调味料和减盐方法', FDA+'nutrition-education-resources-materials/sodium-your-diet', '美国标签标准与参考值；与WHO或中国指南分开标注'),
    ('sugars', '添加糖与总糖：营养标签读法和饮料选择', FDA+'nutrition-facts-label/added-sugars-nutrition-facts-label', '美国FDA标签定义；添加糖不等同游离糖'),
    ('food_safety', '食品安全：清洁、生熟分开、加热与冷藏', FDA+'buy-store-serve-safe-food/safe-food-handling', '食品安全科普；不能凭食物图片或气味证明安全'),
    ('produce', '蔬菜水果安全：选购、清洗、冷藏和交叉污染', FDA+'buy-store-serve-safe-food/selecting-and-serving-produce-safely', '生鲜蔬果家庭食品安全'),
    ('allergens', '食物过敏：常见过敏原、标签与交叉接触', FDA+'nutrition-food-labeling-and-critical-foods/food-allergies', '美国过敏原标签制度；过敏反应需及时就医'),
    ('fish', '鱼类选择：孕妇儿童吃鱼与汞风险', FDA+'consumers/advice-about-eating-fish', '鱼种、份量与适用人群需按原文核对'),
    ('safe_temperatures', '食品安全温度：鸡肉、鱼、绞肉与剩菜加热', 'https://www.foodsafety.gov/food-safety-charts/safe-minimum-internal-temperatures', '美国政府家庭食品安全参考；摄氏华氏和静置时间均按原文核对'),
    ('food_storage', '冰箱储存期限：生肉、熟食与剩菜保存', 'https://www.foodsafety.gov/food-safety-charts/cold-food-storage-charts', '储存期限需同时满足原文冷藏冷冻条件，不能仅凭气味判断安全'),
    ('four_steps', '家庭食品安全四步骤：清洁、分开、加热与冷藏', 'https://www.foodsafety.gov/keep-food-safe/4-steps-to-food-safety', '美国政府家庭食品安全教育'),
]
# Correct the canonical NIDDK pregnancy route before fetching.
SOURCES[6] = (*SOURCES[6][:2], NIDDK+'weight-management/health-tips-pregnant-women', SOURCES[6][3])


class MainText(HTMLParser):
    """Keep main text and table-cell separators, excluding navigation and active content."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.output = []
        self.suppressed = 0
        self.in_main = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        hidden = tag in {'script', 'style', 'nav', 'aside', 'header', 'footer', 'form', 'button', 'svg', 'noscript'}
        hidden |= attrs.get('aria-hidden') == 'true'
        is_main = tag == 'main' or attrs.get('role') == 'main'
        if tag in {'br', 'img', 'input', 'hr', 'meta', 'link', 'source', 'wbr'}:
            if self.in_main and not self.suppressed and tag in {'br', 'hr'}:
                self.output.append('\n')
            return
        self.stack.append((tag, hidden, is_main))
        self.suppressed += hidden
        self.in_main += is_main
        if self.in_main and not self.suppressed:
            if tag in {'p','div','section','article','h1','h2','h3','h4','li','tr'}:
                self.output.append('\n')
            if tag == 'li':
                self.output.append('- ')
            if tag in {'td','th'}:
                self.output.append(' | ')

    def handle_endtag(self, tag):
        indices = [i for i, item in enumerate(self.stack) if item[0] == tag]
        if not indices:
            return
        i = indices[-1]
        popped, self.stack = self.stack[i:], self.stack[:i]
        for _, hidden, main in popped:
            self.suppressed -= hidden
            self.in_main -= main
        if tag in {'p','div','li','tr','h1','h2','h3','h4','section'}:
            self.output.append('\n')

    def handle_data(self, text):
        if self.in_main and not self.suppressed:
            self.output.append(text)

    def text(self):
        lines = [re.sub(r'\s+', ' ', s).strip() for s in ''.join(self.output).splitlines()]
        return '\n'.join(line for line in lines if line)


def fetch(url):
    response = requests.get(url, timeout=(10, 40), headers={'User-Agent': 'NutritionKnowledgeImporter/1.0'})
    response.raise_for_status()
    return response


def acquire(item):
    name, title, url, scope = item
    try:
        response = fetch(url)
        parser = MainText()
        parser.feed(response.text)
        text = parser.text()
        # Drop chrome before the actual page title when the site has nested layout regions.
        if len(text) < 800 or not re.search(r'[A-Za-z]{5}', text):
            raise ValueError('Missing or insufficient article body')
        if re.search(r'(access denied|enable javascript to continue|page not found)', text[:300], re.I):
            raise ValueError('Error page received')
        if re.search(r'copyright\s*(?:©|\(c\))', text, re.I):
            raise ValueError('Third-party copyright notice requires manual review')
        agency = 'NIH NIDDK' if 'niddk.nih.gov' in url else 'US HHS FoodSafety.gov' if 'foodsafety.gov' in url else 'FDA'
        review = re.search(r'Last Reviewed\s+([A-Za-z]+\s+\d{4})', text)
        meta = {'title': title, 'publisher': agency, 'source_url': response.url, 'reviewed_at': TODAY,
                'language': 'en; zh topic heading', 'scope': scope,
                'published_at': review.group(1) if review else '',
                'rights': 'US federal agency original public-domain text; no logos or images reused',
                'rights_policy': NIH_POLICY if agency.startswith('NIH') else FDA_POLICY}
        content = (f'# {title}\n\n适用范围：{scope}。\n'
                   f'以下为机构英文正文，保留原始单位和术语，不是中文逐字翻译。\n'
                   f'来源：{agency}\n原文：{response.url}\n采集核验：{TODAY}\n\n'
                   f'## Original article\n\n{text}\n')
        return f'guides/{name}.md', content, meta, None
    except Exception as error:
        return name, None, None, type(error).__name__


def norm(text):
    text = unicodedata.normalize('NFKC', text).lower()
    return re.sub(r'[^a-z0-9]', '', text)


def cookbook():
    url = 'https://www.nhlbi.nih.gov/sites/default/files/publications/10-2921.pdf'
    cache = ROOT.parent / '.runtime/nhlbi-cookbook-import.pdf'
    data = cache.read_bytes() if cache.exists() else fetch(url).content
    if not data.startswith(b'%PDF'):
        raise ValueError('Cookbook download is not a PDF')
    logging.getLogger('pypdf').setLevel(logging.ERROR)
    reader = PdfReader(BytesIO(data))
    pages = [page.extract_text() or '' for page in reader.pages]
    excluded = {'main dishes','beef','poultry','seafood','pork','main-dish pastas',
                'vegetarian main-dish meals','side dishes','vegetable side dishes','grain side dishes'}
    entries = {}
    for page in pages[:10]:
        for title, number in re.findall(r'(?m)^\s*(.+?)\s*\.{3,}\s*(\d+)\s*$', page):
            title = re.sub(r'\s+', ' ', title).strip()
            n = int(number)
            if 5 <= n <= 126 and title.lower() not in excluded:
                entries[n] = title
    if len(entries) < 65:
        raise ValueError(f'Incomplete recipe contents: {len(entries)}')
    first_n = min(entries)
    first_title = entries[first_n]
    matches = [i for i, text in enumerate(pages) if i > 9 and norm(first_title) in norm(text)
               and re.search(r'yield|serving|calori', text, re.I)]
    if not matches:
        raise ValueError('Cannot align cookbook page numbers')
    offset = matches[0] - first_n
    output = []
    ordered = sorted(entries.items())
    for i, (n, title) in enumerate(ordered):
        stop = ordered[i+1][0] if i+1 < len(ordered) else 128
        start, end = n + offset, stop + offset
        text = '\n\n'.join(f'### 原PDF第 {j+1} 页\n\n{pages[j]}' for j in range(start, end))
        if not re.search(r'yield|serving|calori', text, re.I) or len(text) < 400:
            raise ValueError(f'Incomplete recipe: {title}')
        category = ('牛肉' if n < 17 else '鸡肉与火鸡' if n < 39 else '鱼虾海鲜' if n < 55 else
                    '猪肉' if n < 63 else '意面' if n < 81 else '豆类与素食' if n < 99 else
                    '蔬菜配菜' if n < 115 else '谷物与主食')
        scope = ('2009年官方食谱原方，仅作烹饪参考，不作为最新疾病治疗或营养标准；'
                 '杯、汤匙、盎司等单位和份数按原文，营养值仅对应原方，替换食材后需重新估算')
        meta = {'title': f'官方菜谱｜{category}｜{title}', 'publisher': 'NIH NHLBI',
                'source_url': url+f'#page={start+1}', 'reviewed_at': TODAY, 'published_at': '2009-10',
                'language': 'en; zh category heading', 'scope': scope,
                'rights': 'US federal NIH publication text only; images and trademarks excluded',
                'rights_policy': NIH_POLICY, 'source_pdf_pages': [start+1, end]}
        name = f'recipes/nhlbi_{n:03d}_'+re.sub(r'[^a-z0-9]+','_',title.lower()).strip('_')+'.md'
        output.append((name, f'# {meta["title"]}\n\n适用范围：{scope}。\n'
                       f'原文：{meta["source_url"]}\n\n## 食材、做法及原方营养信息\n\n{text}\n', meta))
    return output


def main():
    runtime = ROOT.parent / '.runtime'
    destination = runtime / ('rag-expansion-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    docs = destination / 'documents'
    docs.mkdir(parents=True)
    rows, previous = knowledge_store._read()
    storage = knowledge_store.storage
    if storage is None:
        raise ValueError('This importer expects the current knowledge to be in COS')
    metadata = {}
    report = {'previous_version': previous['version'], 'staging': str(destination),
              'reviewed_at': TODAY, 'failed_sources': [], 'articles': 0, 'recipes': 0}
    def save(name, content, meta):
        path = docs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode('utf-8'))
        metadata[name] = meta
    for name, data, meta in storage.read_documents(previous['dataset_id']):
        save(name, data, meta)
    print('Preserved existing documents:', len(metadata), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, text, meta, error in pool.map(acquire, SOURCES):
            if error:
                report['failed_sources'].append({'name': name, 'error': error})
                print('Skipped:', name, error, flush=True)
            else:
                save(name, text, meta)
                report['articles'] += 1
                print('Article:', name, 'characters:',len(text),flush=True)
    recipes = cookbook()
    for name, text, meta in recipes:
        save(name, text, meta)
    report['recipes'] = len(recipes)
    report['documents'] = len(metadata)
    report['characters'] = sum(len(p.read_text(encoding='utf-8')) for p in docs.rglob('*.md'))
    (destination/'sources.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    (runtime/'knowledge-expansion-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    assert report['articles'] >= 8 and report['recipes'] >= 65
    print('Prepared:',json.dumps(report,ensure_ascii=True),flush=True)


if __name__ == '__main__':
    main()
