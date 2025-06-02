# 要件とタスクリスト
実行時はまずを以下を読み込み内容を把握する事

bank/basic_design_document.md
bank/todos_overview.yaml
bank/implement_steps.yaml




# 計画の立案
- 実行する task を把握し, まずは計画を立案する事



# 開発コマンド
- uv を用いる事

# ブランチ戦略
- グローバルCLAUDE.mdのbranch_strategyに従う
- developブランチを開発のメインブランチとして使用



# コードの修正後
- document を修正内容を反映すること.
    - 不要な部分は適宜削除すること.
- test に修正内容を反映する事.
- 以下を実行し, 正常であるか確認する事
    - `uv run python -m pytest tests/ -v`
    - `uv run python -m black .`
    - `uv run ruff check claude_code_cost_collector/ tests/`
    - `uv run ruff format claude_code_cost_collector/ tests/` (import整理含む)
    - `uv run python -m mypy claude_code_cost_collector/`



# テスト関連
## 結合テスト (Integration Tests)
- 場所: `tests/integration/`
- 実行: `uv run python -m pytest tests/integration/ -v`
- 46のテストでエンドツーエンド動作を検証
- テストデータ: `tests/integration/test_data/`（総コスト$0.241、総トークン6,900）
- カバー範囲: 全出力形式、集計単位、日付フィルタ、エラーハンドリング
- 詳細は `tests/integration/README.md` を参照

## ユニットテスト
- 場所: `tests/`
- 実行: `uv run python -m pytest tests/ -v`
- 各モジュールの個別機能をテスト

## テスト実行のベストプラクティス
- 新機能追加時は対応する結合テストも追加
- テストデータ変更時は期待値も更新
- エラーケースのテストも忘れずに実装

## 包括的テスト観点
新機能実装時は以下の観点で多角的テストを実施する:

### 1. 基本機能テスト
- 各オプション・パラメータの単体動作確認
- 有効値・無効値の境界値テスト
- エラーメッセージとヘルプ表示の確認

### 2. マトリックステスト
- 機能オプション × granularity の全組み合わせ
- 機能オプション × 出力フォーマット の全組み合わせ
- 複数オプション同時指定での相互作用確認

### 3. 他機能との組み合わせテスト
- --limit, --start-date, --end-date との組み合わせ
- --currency, --timezone との組み合わせ
- 複数フィルタ・ソート・制限の処理順序確認

### 4. エッジケース・境界値テスト
- 空データ・単一データでの動作
- 極端な値（limit 0, 巨大limit）での安全性
- 同値データでのソート安定性
- データフィルタ結果が空の場合

### 5. 国際化・環境依存テスト
- 複数通貨での動作確認
- タイムゾーン指定での正確性
- 無効な国際化パラメータへの堅牢性

### 6. 出力一貫性テスト  
- 全フォーマット（text/json/yaml/csv）での結果順序一致
- メタデータ・サマリ情報の正確性
- 文字エンコーディングの確認

### 7. パフォーマンス・安定性テスト
- 実行時間への影響測定（time コマンド使用）
- メモリ使用量の確認
- 大量データでのスケーラビリティ

### 8. 回帰テスト
- 既存の全テストスイート実行（244個）
- コード品質チェック（black, ruff, mypy）
- 既存機能への影響がないことの確認

# 開発ツール情報
## コード品質ツール
- **linter**: ruff (旧flake8から移行)
- **formatter**: black + ruff format (import整理)
- **type checker**: mypy
- **test runner**: pytest
- **package manager**: uv

## ruff設定
- line-length: 130 (blackと統一)
- target-version: py313
- 有効ルール: E (pycodestyle errors), F (pyflakes), I (isort)
- import整理: known-first-party設定でプロジェクトモジュール識別
- split-on-trailing-comma: blackとの互換性確保

## 開発ワークフロー
1. コード修正
2. `uv run ruff format` - import整理含むフォーマット
3. `uv run ruff check` - リンティング
4. `uv run python -m black .` - 最終フォーマット確認
5. `uv run python -m mypy` - 型チェック
6. `uv run python -m pytest tests/ -v` - テスト実行

# Release作業手順
## バージョン更新対象ファイル
1. `pyproject.toml` - version フィールド (現在: 1.0.0)
2. `claude_code_cost_collector/__init__.py` - __version__ 変数
3. `tests/integration/test_end_to_end.py` - バージョンテストの期待値
4. `CHANGELOG.md` - 新バージョンのエントリ追加

**重要**: バージョン変更時は上記4ファイル全ての更新が必要です

## Release手順
### 事前準備
1. 最新のdevelop branchに移動
   - `git checkout develop`
   - `git pull origin develop`

2. release branch作成
   - `git checkout -b release/v{X.Y.Z}`

### バージョン更新作業
3. バージョン番号の統一確認
   - pyproject.toml と __init__.py のバージョンが一致していること
4. 説明文の言語統一
   - __init__.py の description を英語に統一
5. CHANGELOG.md の更新
   - 新バージョン `## [X.Y.Z] - YYYY-MM-DD` セクションを追加
   - `### Added`, `### Changed`, `### Fixed`, `### Documentation` セクションで変更内容を整理
   - 具体的な変更内容を記述（ユーザー視点での利点を含む）

### 検証
6. テスト実行
   - `uv run python -m pytest tests/ -v`
   - `uv run python -m black .`
   - `uv run ruff check claude_code_cost_collector/ tests/`
   - `uv run ruff format claude_code_cost_collector/ tests/`
   - `uv run python -m mypy claude_code_cost_collector/`
7. ビルドとパッケージング
   - `uv build`

### Git操作とマージ
8. Release branch のコミット
   - `git add .`
   - `git commit -m "chore: bump version to {X.Y.Z}"`
   - `git push origin release/v{X.Y.Z}`

9. develop branch への統合
   - `git checkout develop`
   - `git merge --no-ff release/v{X.Y.Z}`
   - `git push origin develop`

10. main branch への統合
    - `git checkout main`
    - `git pull origin main`
    - `git merge --no-ff develop`
    - `git push origin main`

11. タグ付けとプッシュ
    - `git tag v{X.Y.Z}`
    - `git push origin v{X.Y.Z}`

12. Release branch削除
    - `git branch -d release/v{X.Y.Z}`
    - `git push origin --delete release/v{X.Y.Z}`
