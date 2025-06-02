# 開発者ガイド - Claude Code Cost Collector

## 概要

このドキュメントは、Claude Code Cost Collector (CCCA) の開発者向け詳細ガイドです。アーキテクチャの理解、開発プロセス、デバッグ方法、拡張方法について説明します。

## アーキテクチャ概要

### システム全体の処理フロー

```
CLI引数解析 → データ収集 → データ解析 → データ集計 → 出力処理
    ↓           ↓          ↓          ↓         ↓
  cli.py → collector.py → parser.py → aggregator.py → formatter.py
```

### モジュール構成

#### 1. **CLI引数パーサー** (`cli.py`)
- **役割**: コマンドライン引数の解析と検証
- **主要関数**: `parse_args()`
- **依存関係**: `argparse` (標準ライブラリ)

```python
def parse_args() -> argparse.Namespace:
    \"\"\"Parse command line arguments.\"\"\"
    # 実装詳細...
```

#### 2. **データモデル** (`models.py`)
- **役割**: 共通データ構造の定義
- **主要クラス**: `ProcessedLogEntry`
- **設計原則**: 不変性、型安全性

```python
@dataclass
class ProcessedLogEntry:
    timestamp: datetime
    date_str: str
    month_str: str
    project_name: str
    session_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    model: str
```

#### 3. **ログファイル収集** (`collector.py`)
- **役割**: 指定されたディレクトリから対象JSONファイルの収集
- **主要関数**: `collect_log_files()`
- **使用技術**: `pathlib`, `os.walk`

#### 4. **ログデータ解析** (`parser.py`)
- **役割**: JSONログファイルの解析と`ProcessedLogEntry`への変換
- **主要関数**: `parse_log_file()`, `parse_log_entry()`
- **エラーハンドリング**: 不正なJSONファイルや欠損フィールドの処理

#### 5. **データ集計** (`aggregator.py`)
- **役割**: 解析済みデータの集計処理
- **集計単位**: 日次、月次、プロジェクト別、セッション別
- **主要関数**: `aggregate_data()`

#### 6. **出力フォーマット** (`formatter.py`)
- **役割**: 集計結果の各種形式での出力
- **対応形式**: Text Table, JSON, YAML, CSV
- **使用ライブラリ**: `rich`, `json`, `PyYAML`, `csv`


#### 8. **設定ファイル処理** (`config.py`)
- **役割**: 設定ファイルの読み書き
- **設定項目**: APIキー、デフォルト値等

#### 9. **メインオーケストレーター** (`main.py`)
- **役割**: 全体の処理フローの制御
- **責務**: モジュール間の調整、エラー処理、ログ出力

## 詳細設計

### データフロー詳細

1. **引数解析フェーズ**
   ```python
   args = cli.parse_args()
   # args.directory, args.granularity, args.output 等が設定される
   ```

2. **設定読み込みフェーズ**
   ```python
   config_data = config.load_config(args.config) if args.config else {}
   ```

3. **データ収集フェーズ**
   ```python
   log_files = collector.collect_log_files(args.directory)
   # List[Path] が返される
   ```

4. **データ解析フェーズ**
   ```python
   processed_entries = []
   for log_file in log_files:
       entries = parser.parse_log_file(log_file)
       processed_entries.extend(entries)
   # List[ProcessedLogEntry] が構築される
   ```

5. **データ集計フェーズ**
   ```python
   aggregated_data = aggregator.aggregate_data(
       processed_entries, 
       granularity=args.granularity
   )
   # 集計結果がDict形式で返される
   ```

6. **出力フェーズ**
   ```python
   formatter.format_output(
       aggregated_data, 
       output_format=args.output
   )
   ```

### エラーハンドリング戦略

#### 1. **ファイル処理エラー**
- 存在しないディレクトリ: 適切なエラーメッセージと終了
- 読み取り権限なし: スキップしてログ出力
- 破損したJSONファイル: スキップして警告出力

#### 2. **API関連エラー**
- 為替レートAPI障害: デフォルト値または処理スキップ
- ネットワーク接続エラー: タイムアウト設定とリトライ

#### 3. **データ不整合**
- 必須フィールド欠損: ログ出力してスキップ
- 型不一致: 可能な範囲で型変換、不可能な場合はスキップ

### ログ出力設計

```python
import logging

# ログレベル設定
DEBUG: 詳細なデバッグ情報
INFO:  処理進捗情報
WARNING: 処理続行可能な問題
ERROR: 処理停止を伴う問題
CRITICAL: システム全体に影響する問題
```

## 開発プロセス

### 1. **新機能開発の流れ**

1. **要件分析**
   - `bank/basic_design_document.md` で要件確認
   - 影響範囲の特定

2. **設計検討**
   - 既存アーキテクチャとの整合性確認
   - インターフェース設計
   - テスト設計

3. **実装**
   - 小さな単位で実装・テスト
   - 定期的なコミット
   - テストファースト開発の推奨

4. **統合テスト**
   - 全体の動作確認
   - パフォーマンステスト
   - 回帰テスト

### 2. **テスト戦略**

#### ユニットテスト
- 各モジュールの公開関数に対してテストケースを作成
- モックオブジェクトを使用して外部依存を排除
- エッジケース、エラーケースを網羅

#### 統合テスト
- モジュール間の連携テスト
- 実際のファイルシステムを使用したテスト
- E2Eシナリオのテスト

#### テストデータ管理
```
tests/
├── fixtures/           # テスト用固定データ
│   ├── sample_logs/   # サンプルログファイル
│   └── config/        # テスト用設定ファイル
└── test_*.py          # テストファイル
```

### 3. **パフォーマンス最適化**

#### 大量ファイル処理
- ジェネレータの活用によるメモリ効率化
- 並列処理の検討 (multiprocessing)
- プログレス表示の実装

#### メモリ使用量最適化
- 不要なデータの早期解放
- ストリーミング処理の活用
- メモリプロファイリングツールの使用

## デバッグガイド

### 1. **ログレベル設定**

```bash
# デバッグ情報を有効化
export PYTHONPATH=$PWD
python -m claude_code_cost_collector.main --debug
```

### 2. **よくある問題と対処法**

#### **問題**: ログファイルが見つからない
```bash
# ディレクトリの存在確認
ls -la ~/.claude/projects/

# 権限確認
ls -la ~/.claude/projects/*/*.json

# デバッグモードで詳細確認
ccc --directory ~/.claude/projects --debug
```

#### **問題**: JSON解析エラー
```bash
# 問題のあるJSONファイルの特定
find ~/.claude/projects -name "*.json" -exec python -m json.tool {} \\; > /dev/null
```

#### **問題**: 為替レート取得失敗
```bash
# ネットワーク接続確認
curl -I https://api.exchangerate-api.com/v4/latest/USD

# APIキーの確認
echo $EXCHANGE_API_KEY
```

### 3. **プロファイリング**

```python
# パフォーマンス測定
import cProfile
import pstats

cProfile.run('main()', 'profile_output')
stats = pstats.Stats('profile_output')
stats.sort_stats('cumulative').print_stats(10)
```

## 拡張ガイド

### 1. **新しい集計単位の追加**

1. **aggregator.pyの修正**
   ```python
   def aggregate_by_custom_unit(entries: List[ProcessedLogEntry]) -> Dict[str, Any]:
       \"\"\"Custom aggregation logic.\"\"\"
       # 実装...
   ```

2. **CLI引数の追加**
   ```python
   # cli.pyに新しいオプション追加
   parser.add_argument('--granularity', choices=['daily', 'monthly', 'project', 'session', 'custom', 'all'])
   parser.add_argument('--sort-field', choices=['input', 'output', 'total', 'cost', 'date', 'custom'])
   ```

3. **テストケースの追加**
   ```python
   def test_aggregate_by_custom_unit():
       # テスト実装...
   ```

### 2. **新しい出力形式の追加**

1. **formatter.pyの修正**
   ```python
   def format_custom_output(data: Dict[str, Any], **kwargs) -> str:
       \"\"\"Custom output formatting.\"\"\"
       # 実装...
   ```

2. **メイン処理への統合**
   ```python
   # formatter.pyのdispatch機構に追加
   FORMAT_HANDLERS = {
       'text': format_text_table,
       'json': format_json,
       'yaml': format_yaml,
       'csv': format_csv,
       'custom': format_custom_output,  # 新規追加
   }
   ```

### 3. **新しいデータソースの対応**

1. **collector.pyの拡張**
   ```python
   def collect_from_new_source(source_config: Dict[str, Any]) -> List[Path]:
       \"\"\"New data source collector.\"\"\"
       # 実装...
   ```

2. **parser.pyの拡張**
   ```python
   def parse_new_format(file_path: Path) -> List[ProcessedLogEntry]:
       \"\"\"New format parser.\"\"\"
       # 実装...
   ```

### 4. **プラグインシステムの実装**

```python
# plugins/base.py
from abc import ABC, abstractmethod

class OutputPlugin(ABC):
    @abstractmethod
    def format_output(self, data: Dict[str, Any]) -> str:
        pass

# plugins/example_plugin.py
class ExamplePlugin(OutputPlugin):
    def format_output(self, data: Dict[str, Any]) -> str:
        # カスタム実装...
        return formatted_output
```

## 開発ツール

### 1. **推奨エディタ設定 (VS Code)**

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "ruff.enable": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"]
}
```

### 2. **開発用スクリプト**

```bash
#!/bin/bash
# scripts/dev-setup.sh
echo "Setting up development environment..."
uv sync --group dev
source .venv/bin/activate
echo "Running initial tests..."
pytest
echo "Setup complete!"
```

### 3. **継続的インテグレーション**

```yaml
# .github/workflows/ci.yml の例
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.13]
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install uv
        uv sync --group dev
    - name: Run tests
      run: |
        source .venv/bin/activate
        pytest --cov=claude_code_cost_collector
    - name: Run linting
      run: |
        source .venv/bin/activate
        ruff check claude_code_cost_collector/ tests/
        ruff format --check claude_code_cost_collector/ tests/
        black --check .
        mypy claude_code_cost_collector/
```

## リリース手順

### 1. **バージョン管理**
- セマンティックバージョニング (MAJOR.MINOR.PATCH)
- `pyproject.toml` のバージョン更新
- Git タグの作成

### 2. **チェンジログ生成**
```bash
# CHANGELOG.md の更新
git log --oneline --no-merges v0.1.0..HEAD >> CHANGELOG.md
```

### 3. **リリースビルド**
```bash
# パッケージビルド
uv build

# 成果物確認
ls dist/
```

## トラブルシューティング

### 開発環境の問題

1. **モジュールが見つからない**
   ```bash
   # PYTHONPATH の確認
   echo $PYTHONPATH
   
   # パッケージの再インストール
   pip install -e .
   ```

2. **テストが失敗する**
   ```bash
   # 詳細な出力でテスト実行
   pytest -v -s
   
   # 特定のテストのみ実行
   pytest tests/test_parser.py::test_specific_function -v
   ```

3. **型チェックエラー**
   ```bash
   # 詳細な型エラー情報
   mypy claude_code_cost_collector/ --show-error-codes
   ```

このガイドは開発の進行とともに継続的に更新されます。新しい機能や設計変更があった場合は、このドキュメントも同時に更新してください。