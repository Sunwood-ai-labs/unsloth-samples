#!/bin/bash
# Unsloth LLMファインチューニング環境のセットアップスクリプト

echo "🚀 Unsloth環境をセットアップしています..."

# uvがインストールされているか確認
if ! command -v uv &> /dev/null; then
    echo "📦 uvをインストールしています..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "📚 依存関係をインストールしています..."
uv sync

echo "✅ セットアップが完了しました！"
echo ""
echo "使い方:"
echo "  1. 仮想環境をアクティベート: source .venv/bin/activate"
echo "  2. サンプルスクリプトを実行: python finetune_example.py"
echo ""
echo "または:"
echo "  uv run python finetune_example.py"
