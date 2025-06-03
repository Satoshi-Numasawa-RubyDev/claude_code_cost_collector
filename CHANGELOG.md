# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] - 2025-06-03

### Changed
- **Smart Default Sorting**: Enhanced `--sort` option behavior for improved user experience
  - When only `--sort asc` or `--sort desc` is specified (without `--sort-field`), automatically defaults to sorting by date
  - Provides intuitive chronological ordering for time-series cost data
  - Maintains backward compatibility when both sort direction and field are explicitly specified
  - Eliminates the need to explicitly specify `--sort-field date` for common date-based sorting use cases

### Documentation
- **README.md Updates**: Updated sort functionality documentation
  - Added examples of smart default sorting behavior (`cccc --sort asc` for date ascending)
  - Updated output examples to show both ascending and descending date sort results
  - Enhanced sort functionality section with clearer explanations of available options
  - Added advanced usage examples demonstrating the new smart sorting feature

## [1.0.2] - 2025-06-02

### Documentation
- **Sort Interface Documentation Update**: Updated all documentation to reflect new sort interface
  - Updated README.md examples to use `--sort desc --sort-field cost` format
  - Updated CONTRIBUTING.md examples and development tool commands
  - Updated docs/developer-guide.md with new interface examples
  - Improved consistency across all documentation files

### Added
- **Development Best Practices**: Added comprehensive development rules to CLAUDE.md
  - Feature deletion/modification residue checking rules with ripgrep commands
  - Standardized comment annotation rules (TODO/FIXME/NOTE/HACK/BUG/OPTIMIZE)
  - Unified date format `<YYYY-MM-DD>` for improved searchability
  - Ownership requirement for all annotations with clear responsibility tracking

### Changed
- **Development Tools Documentation**: Updated all tool references from flake8 to ruff
  - Updated VS Code settings recommendations
  - Updated CI/CD pipeline examples
  - Improved development workflow with unified formatting and linting

### Technical Improvements
- Enhanced development process consistency with standardized practices
- Improved code maintainability through systematic annotation tracking
- Better documentation searchability and maintenance procedures

## [1.0.1] - 2025-06-02

### Changed
- **Development Tools**: Migrated from flake8 to ruff for unified linting and formatting
  - Replaced flake8 with ruff for improved performance and unified tooling
  - Added ruff isort integration for automatic import organization
  - Updated development workflows and documentation to reflect new tooling

### Documentation
- Updated CLAUDE.md with comprehensive development tools information
- Added detailed ruff configuration and workflow documentation
- Corrected test count and version information in project documentation

### Technical Improvements
- Unified linting, formatting, and import organization under ruff
- Improved code consistency with automatic import sorting
- Enhanced development experience with faster tooling

## [1.0.0] - 2025-06-02

### Changed
- **BREAKING CHANGE: Sort Interface Redesign**: Completely redesigned sort command-line interface for better usability
  - `--sort` now specifies sort order (asc/desc) instead of field selection
  - `--sort-field` now specifies which field to sort by (input/output/total/cost/date)
  - Default sort order changed from ascending to descending
  - **Migration**: `--sort cost --sort-desc` → `--sort desc --sort-field cost`
  - **Migration**: `--sort input` → `--sort asc --sort-field input`
  - Removed deprecated `--sort-desc` argument

### Added
- **Date Sorting**: New date sorting capability with `--sort-field date`
  - Sort aggregated data by date in ascending or descending order
  - Works with all granularities (daily, monthly, project, session)
  - Lexicographic sorting for consistent behavior across different key formats

### Fixed
- **Argument Processing**: Improved robustness of sort argument handling
  - Better handling of None values in argument processing
  - Enhanced validation for argument combinations

### Testing
- **Test Suite Optimization**: Comprehensive test suite improvements and optimization
  - Reduced test count from 255 to 244 tests (11 tests removed for efficiency)
  - Removed 17 redundant tests across multiple categories:
    - 5 duplicate empty entry tests across aggregation classes
    - 2 duplicate validation tests 
    - 10 redundant sort functionality tests
  - **Enhanced Error Handling**: Added 6 new robust error handling tests
    - Negative token and cost value processing
    - Extremely large numeric value handling (10^15+ scale)
    - Unicode and special character support in project names (Japanese, Cyrillic, emojis)
    - Zero value processing validation
    - Extremely long string field handling (1000+ characters)
    - Sort stability with identical timestamps
  - **Improved Test Coverage**: Added comprehensive edge case testing
    - Better boundary value testing
    - Enhanced international character support validation
    - Improved sorting algorithm stability verification
  - **Performance Optimization**: Test execution time improved to ~9.8 seconds
  - **Quality Assurance**: Maintained 100% test pass rate (244/244 tests)

### Documentation
- **Interface Migration Guide**: Added migration examples for new sort interface
- **Enhanced Help Text**: Updated help messages to reflect new argument structure and defaults

## [0.5.3] - 2025-06-02

### Changed
- **Repository URL Update**: Updated GitHub repository URL in pyproject.toml
  - Reflects current repository location for better package discovery
  - Ensures accurate source code references in package metadata

### Added
- **Development Branch Strategy**: Established develop branch as main development branch
  - Enhanced development workflow with structured branch management
  - Added branch strategy documentation to CLAUDE.md

## [0.5.2] - 2025-06-02

### Added
- **Package Metadata Enhancement**: Added repository URL to pyproject.toml
  - Improved package discovery and maintenance by providing direct link to source code
  - Enhanced user experience by enabling easy access to documentation and issue reporting
  - Better integration with Python package ecosystem and tooling

## [0.5.1] - 2025-06-01

### Documentation
- **README Screenshot Update**: Updated application screenshot with latest interface changes
  - Refreshed visual documentation to reflect current UI state
  - Improved user onboarding experience with up-to-date visual examples

## [0.5.0] - 2025-06-01

### Documentation
- **Release Process Enhancement**: Comprehensive update to release workflow documentation
  - Added detailed Git Flow with branch strategy (develop → release → main)
  - Clarified step-by-step release procedures including pre-checks, version updates, testing, and merge process
  - Improved maintainability and consistency of future releases
  - Enhanced developer experience with clear documentation of branch operations and cleanup

### Changed
- **Version Bump**: Updated from 0.4.4 to 0.5.0 to reflect improved release process

## [0.4.4] - 2025-05-31

### Removed
- **Code Cleanup**: Removed 5 unused functions to improve maintainability
  - `detect_legacy_config_files()`, `suggest_config_migration()`, `migrate_legacy_config_file()` from config.py
  - `get_cached_rate()`, `clear_cache()` from exchange.py
  - Total reduction: 98 lines of unused code

### Fixed
- **Test Standardization**: Updated error message assertions to use English consistently
  - Replaced Japanese error messages with English equivalents in test files
  - Improved test maintainability and international compatibility
- **Version Test Update**: Updated version test assertion from 0.1.0 to 0.4.4

### Changed
- **Code Quality**: Added setup.cfg for proper flake8 configuration
  - Ensures 130-character line length limit is properly applied
  - Improved development workflow consistency

## [0.4.3] - 2025-05-31

### Fixed
- **バージョン統一**: `claude_code_cost_collector/__init__.py` のバージョンを 0.1.0 から 0.4.3 に更新
  - `pyproject.toml` とバージョン番号を統一
- **言語統一**: `__init__.py` 内の説明文を日本語から英語に変更
  - プロジェクト全体の国際化対応の一環
  - パッケージメタデータの一貫性向上

### Documentation
- **リリース手順の文書化**: `CLAUDE.md` にリリース作業手順を追加
  - バージョン更新対象ファイルの明記
  - 標準化されたリリースプロセスの確立

## [0.4.0] - 2025-05-31

### Changed
- **プロジェクト名の最終化**: プロジェクト名を "claude-code-cost-analyzer-advanced" から "claude-code-cost-analyzer" に変更
  - パッケージ管理の簡素化とユーザビリティ向上
  - 一貫性のあるプロジェクト名でのリリース準備

### Added
- **ビルド・パッケージング支援**: 開発依存関係に build と twine を追加
  - `build>=1.2.2.post1`: モダンなPythonパッケージビルドツール
  - `twine>=6.1.0`: PyPI へのセキュアなパッケージアップロード

### Technical Improvements
- 依存関係管理の最適化とリリースプロセスの標準化
- パッケージング環境の整備

## [0.3.0] - 2025-05-31

### Added
- **ソート機能**: 包括的なデータソート機能を実装
  - 日時、コスト、プロジェクト名による並び替え
  - 昇順・降順の切り替え対応
- **制限機能**: `--limit` オプションによる表示件数制限
  - 適切な日時ソートと組み合わせた最新データ表示
  - 全ての集計単位（daily, monthly, project, session, all）で利用可能
  - JSON、YAML、CSV、テキスト形式全てに対応
- **バージョン表示**: `--version` オプションでバージョン情報表示

### Fixed
- コード品質の改善とテスト信頼性の向上
- プロジェクト名の統一（"Claude Code Cost Collector"に修正）
- テストデータパスの汎用化

### Documentation
- 国際化対応でREADME.mdを英語に翻訳
- Apache License 2.0の追加と包括的なCONTRIBUTING.mdの作成
- ドキュメントの冗長部分削除とフォーマット修正

### Technical Improvements
- 結合テストでの`--limit`機能の包括的テスト追加
- 型チェック（mypy）で全11ファイルの型安全性確認
- コードフォーマット（black）とリンティング（flake8）の品質基準クリア

### Example Usage
```bash
# 最新の5日分のデータを表示
cccc --limit 5

# コストの高いプロジェクト上位3つをJSON形式で表示
cccc -g project --limit 3 -o json

# バージョン確認
cccc --version
```

## [0.2.0] - 2025-05-31

### Added
- **`--limit` オプション**: 出力される結果の件数を制限する機能（初期実装）
- **`--version` オプション**: バージョン情報表示機能（初期実装）

### Changed
- **テストデータの一般化**: ユーザー固有のパスを汎用的な表記に変更
- **コードフォーマット**: blackによる一貫したコードスタイルの適用
- **コード品質向上**: flake8、mypyによる静的解析でのエラー解消

## [0.1.0] - 2025-05-31

### Added
- **コア機能**: Claude API利用コスト集計・表示機能
- **コマンドラインインターフェース**: argparseを使用した豊富なオプション
  - `-d, --directory`: ログディレクトリ指定（デフォルト: `~/.claude/projects/`）
  - `-g, --granularity`: 集計単位（daily, monthly, project, session, all）
  - `-o, --output`: 出力形式（text, json, yaml, csv）
  - `--start-date`, `--end-date`: 期間フィルタ
  - `--config`: 設定ファイル指定

- **集計機能**:
  - 日次集計（デフォルト）
  - 月次集計
  - プロジェクト別集計
  - セッション別集計
  - 個別ログエントリ表示

- **出力形式**:
  - 整形済みテキストテーブル（デフォルト、richライブラリ使用）
  - JSON形式
  - YAML形式
  - CSV形式

- **通貨対応**:
  - USD表示（基本）

- **データ処理機能**:
  - JSON ログファイルの再帰的収集
  - ログデータパーサー（必須フィールド抽出、エラーハンドリング）
  - プロジェクト名の自動決定（ディレクトリ構造ベース）
  - タイムスタンプ変換とフィルタリング

- **設定管理**:
  - YAML設定ファイル対応
  - 環境変数・コマンドライン引数の優先度制御

- **テスト**:
  - 包括的なユニットテスト（全モジュール対応）
  - 結合テスト（46テストケース、エンドツーエンド検証）
  - テストデータ（総コスト$0.241、総トークン6,900）

- **開発支援**:
  - 型注釈とmypy対応
  - black、flake8、ruffによるコード品質管理
  - プロジェクト構造の標準化

### Technical Details
- **Python**: 3.13以上
- **依存関係**: requests, PyYAML, rich
- **アーキテクチャ**: モジュラー設計（cli, collector, parser, aggregator, formatter, exchange, config）
- **パッケージング**: setuptools、uvによる依存関係管理

### Documentation
- 包括的なREADME.md（インストール、使用方法、例）
- 開発者ガイド（アーキテクチャ、テスト、コントリビューション）
- 詳細なdocstring（全モジュール・関数）

### Known Limitations
- 現在、exchangerate-api.comのみ為替レートソースとして対応
- 大量のログファイル処理時の性能最適化は今後の課題
- Windowsでのパス処理は基本対応（詳細テストは未実施）

### Example Usage
```bash
# 基本的な日次集計（テキスト形式）
cccc

# プロジェクト別、JSON出力
cccc -g project -o json

# 期間指定、月次集計、CSV出力
cccc -g monthly --start-date 2025-05-01 --end-date 2025-05-31 -o csv
```