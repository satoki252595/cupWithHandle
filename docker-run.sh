#!/bin/bash
# Cup with Handle Pattern Detector - Docker Helper Script
# ========================================================
#
# Usage:
#   ./docker-run.sh build          # Build Docker image
#   ./docker-run.sh scan 7203      # Scan specific stock
#   ./docker-run.sh sample         # Run sample test
#   ./docker-run.sh ml 7203        # ML prediction mode
#   ./docker-run.sh train          # Train ML models
#   ./docker-run.sh backtest       # Run backtest
#   ./docker-run.sh shell          # Interactive shell

set -e

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ヘルプ表示
show_help() {
    echo "Cup with Handle Pattern Detector - Docker Helper"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  build              Build Docker image"
    echo "  scan <symbols>     Scan specific stocks (e.g., scan 7203 6758)"
    echo "  sample [type]      Run sample test (nikkei/semiconductor/growth)"
    echo "  ml <symbols>       ML-enhanced prediction"
    echo "  full               Full screening (all Japanese stocks)"
    echo "  train              Train ML models"
    echo "  backtest           Run backtest"
    echo "  evaluate           Evaluate ML models"
    echo "  shell              Interactive bash shell"
    echo "  logs               Show recent logs"
    echo "  clean              Remove containers and images"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 scan 7203 6758 8035"
    echo "  $0 sample nikkei"
    echo "  $0 ml 7203 --visualize"
    echo "  $0 train"
}

# ビルド
build() {
    echo -e "${GREEN}Building Docker image...${NC}"
    docker-compose build
    echo -e "${GREEN}Build complete!${NC}"
}

# スキャン
scan() {
    if [ $# -eq 0 ]; then
        echo -e "${RED}Error: Please specify stock symbols${NC}"
        echo "Usage: $0 scan 7203 6758 8035"
        exit 1
    fi
    echo -e "${GREEN}Scanning stocks: $@${NC}"
    docker-compose run --rm scanner "$@"
}

# サンプルテスト
sample() {
    local sample_type=${1:-nikkei}
    echo -e "${GREEN}Running sample test: ${sample_type}${NC}"
    docker-compose run --rm scanner --sample "$sample_type"
}

# ML予測
ml_predict() {
    if [ $# -eq 0 ]; then
        echo -e "${RED}Error: Please specify stock symbols${NC}"
        echo "Usage: $0 ml 7203 6758"
        exit 1
    fi
    echo -e "${GREEN}ML prediction for: $@${NC}"
    docker-compose run --rm scanner --ml-mode "$@"
}

# 全銘柄スクリーニング
full_scan() {
    echo -e "${YELLOW}Starting full screening (this may take a while)...${NC}"
    docker-compose run --rm scanner
}

# ML訓練
train() {
    echo -e "${GREEN}Training ML models...${NC}"
    docker-compose run --rm trainer "$@"
}

# バックテスト
backtest() {
    echo -e "${GREEN}Running backtest...${NC}"
    docker-compose run --rm backtest
}

# 評価
evaluate() {
    echo -e "${GREEN}Evaluating ML models...${NC}"
    docker-compose run --rm evaluator
}

# シェル
shell() {
    echo -e "${GREEN}Starting interactive shell...${NC}"
    docker-compose run --rm shell
}

# ログ表示
logs() {
    docker-compose logs -f
}

# クリーンアップ
clean() {
    echo -e "${YELLOW}Removing containers and images...${NC}"
    docker-compose down --rmi local -v
    echo -e "${GREEN}Cleanup complete!${NC}"
}

# メイン処理
case "${1:-help}" in
    build)
        build
        ;;
    scan)
        shift
        scan "$@"
        ;;
    sample)
        shift
        sample "$@"
        ;;
    ml)
        shift
        ml_predict "$@"
        ;;
    full)
        full_scan
        ;;
    train)
        shift
        train "$@"
        ;;
    backtest)
        backtest
        ;;
    evaluate)
        evaluate
        ;;
    shell)
        shell
        ;;
    logs)
        logs
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac
