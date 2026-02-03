# 测试路由是否正确注册

from src.app import create_app

app = create_app()

print("所有路由：")
for rule in app.url_map.iter_rules():
    if 'character' in rule.rule.lower() and 'image' in rule.rule.lower():
        print(f"  {rule.rule} -> {rule.methods}")

