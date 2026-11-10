"""Deterministic ingredient aliases for Chinese queries over English source recipes."""
ALIASES = {
    '西兰花': ('broccoli',), '花椰菜': ('cauliflower',), '菜花': ('cauliflower',),
    '豆腐': ('tofu',), '三文鱼': ('salmon',), '鲑鱼': ('salmon',),
    '鳕鱼': ('cod',), '金枪鱼': ('tuna',), '虾': ('shrimp',), '扇贝': ('scallops',),
    '鸡肉': ('chicken',), '鸡胸': ('chicken', 'breast'), '火鸡': ('turkey',),
    '牛肉': ('beef',), '猪肉': ('pork',), '鸡蛋': ('egg', 'eggs'),
    '小扁豆': ('lentils', 'lentil'), '鹰嘴豆': ('chickpeas', 'chickpea'),
    '黑豆': ('black', 'beans'), '毛豆': ('edamame',), '红豆': ('red', 'beans'),
    '糙米': ('brown', 'rice'), '藜麦': ('quinoa',), '燕麦': ('oats', 'oatmeal'),
    '羽衣甘蓝': ('kale',), '菠菜': ('spinach',), '胡萝卜': ('carrots', 'carrot'),
    '彩椒': ('pepper',), '甜椒': ('pepper',), '番茄': ('tomato', 'tomatoes'),
    '西红柿': ('tomato', 'tomatoes'), '蘑菇': ('mushroom', 'mushrooms'),
    '香菇': ('shiitake', 'mushroom'), '姜': ('ginger',), '葱': ('scallions', 'onion'),
    '芦笋': ('asparagus',), '南瓜': ('squash',), '玉米': ('corn',),
    '酸奶': ('yogurt',), '花生': ('peanut', 'peanuts'), '芝麻': ('sesame',),
}


def ingredient_terms(query: str) -> set[str]:
    return {term for name, terms in ALIASES.items() if name in query for term in terms}
