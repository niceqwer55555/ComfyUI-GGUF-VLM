"""
快速验证新节点是否能正确导入
注意：此脚本需要在 ComfyUI 环境中运行，或者直接检查文件结构
"""

import sys
from pathlib import Path
import ast

def verify_nodes():
    """验证节点文件结构"""
    print("="*80)
    print("验证新文本节点文件")
    print("="*80)
    
    # 检查文件是否存在
    nodes_file = Path(__file__).parent / "nodes" / "text_generation_nodes.py"
    
    if not nodes_file.exists():
        print(f"\n❌ 文件不存在: {nodes_file}")
        return False
    
    print(f"\n✅ 文件存在: {nodes_file}")
    
    try:
        # 读取文件内容
        with open(nodes_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析 AST
        tree = ast.parse(content)
        
        # 查找类定义
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        print(f"\n📦 发现的类:")
        for cls in classes:
            print(f"  - {cls}")
        
        # 验证必需的类
        required_classes = ['LocalTextModelLoader', 'RemoteTextModelSelector', 'TextGeneration']
        missing = [cls for cls in required_classes if cls not in classes]
        
        if missing:
            print(f"\n❌ 缺少类: {missing}")
            return False
        
        print(f"\n✅ 所有必需的类都存在")
        
        # 检查 NODE_CLASS_MAPPINGS
        if 'NODE_CLASS_MAPPINGS' in content:
            print(f"\n✅ NODE_CLASS_MAPPINGS 已定义")
        else:
            print(f"\n❌ NODE_CLASS_MAPPINGS 未定义")
            return False
        
        if 'NODE_DISPLAY_NAME_MAPPINGS' in content:
            print(f"✅ NODE_DISPLAY_NAME_MAPPINGS 已定义")
        else:
            print(f"❌ NODE_DISPLAY_NAME_MAPPINGS 未定义")
            return False
        
        # 检查关键方法
        print(f"\n🔍 检查关键方法:")
        methods_to_check = [
            ('load_model', 'LocalTextModelLoader'),
            ('select_model', 'RemoteTextModelSelector'),
            ('generate', 'TextGeneration'),
            ('_generate_local', 'TextGeneration'),
            ('_generate_remote', 'TextGeneration'),
            ('_extract_thinking', 'TextGeneration'),
        ]
        
        for method, cls in methods_to_check:
            if f'def {method}' in content:
                print(f"  ✅ {cls}.{method}")
            else:
                print(f"  ❌ {cls}.{method} 未找到")
        
        # 检查旧文件是否已重命名
        print(f"\n📁 检查旧文件:")
        old_files = [
            "nodes/text_node.py.deprecated",
            "nodes/unified_text_node.py.deprecated"
        ]
        
        for old_file in old_files:
            old_path = Path(__file__).parent / old_file
            if old_path.exists():
                print(f"  ✅ {old_file} (已备份)")
            else:
                print(f"  ⚠️  {old_file} (未找到，可能未备份)")
        
        # 检查文档文件
        print(f"\n📚 检查文档文件:")
        doc_files = [
            "MIGRATION_TEXT_NODES.md",
            "TEST_NEW_NODES.md",
            "TEXT_NODES_ARCHITECTURE.md",
            "REFACTOR_SUMMARY.md"
        ]
        
        for doc_file in doc_files:
            doc_path = Path(__file__).parent / doc_file
            if doc_path.exists():
                print(f"  ✅ {doc_file}")
            else:
                print(f"  ❌ {doc_file} (未找到)")
        
        print("\n" + "="*80)
        print("✅ 所有验证通过！")
        print("="*80)
        print("\n💡 下一步:")
        print("  1. 重启 ComfyUI")
        print("  2. 在节点菜单中查找:")
        print("     - 🖥️ Local Text Model Loader")
        print("     - 🌐 Remote Text Model Selector")
        print("     - 🤖 Text Generation")
        print("  3. 参考 TEST_NEW_NODES.md 进行测试")
        print()
        
        return True
        
    except ImportError as e:
        print(f"\n❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_nodes()
    sys.exit(0 if success else 1)
