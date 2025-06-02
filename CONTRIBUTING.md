# Contributing to Claude Code Cost Collector (CCCC)

Claude Code Cost Collector (CCCC) への貢献をご検討いただき、ありがとうございます！このガイドでは、プロジェクトへの貢献方法について説明します。

## 開発環境のセットアップ

### 前提条件

- Python 3.13以上
- uvパッケージマネージャー（推奨）または pip

### 環境構築

1. リポジトリをクローンします：

```bash
git clone <repository-url>
cd claude_code_cost_collector
```

2. 開発依存関係をインストールします：

```bash
# uvを使用（推奨）
uv sync

# またはpipを使用
pip install -e ".[dev]"
```

## 開発ワークフロー

### コード品質の確認

プルリクエストを作成する前に、以下のコマンドでコード品質を確認してください：

```bash
# テストの実行
uv run python -m pytest

# コードフォーマット
uv run ruff format claude_code_cost_collector/ tests/
uv run python -m black .

# リンティング
uv run ruff check claude_code_cost_collector/ tests/

# 型チェック
uv run python -m mypy claude_code_cost_collector
```

### テストの実行

```bash
# 全テストの実行
uv run python -m pytest

# 特定のテストファイルの実行
uv run python -m pytest tests/test_aggregator.py

# カバレッジ付きでテストを実行
uv run python -m pytest --cov=claude_code_cost_collector

# 結合テストの実行
uv run python -m pytest tests/integration/ -v
```

### コーディング規約

- **PEP 8**: Pythonコーディング規約に従ってください
- **型ヒント**: すべての公開関数に型ヒントを追加してください
- **ドキュメント**: 関数とクラスにはdocstringを記述してください
- **テスト**: 新機能には対応するテストケースを追加してください

### ブランチ戦略

- `main`: 本番リリース用ブランチ
- `develop`: 開発統合ブランチ
- `feature/*`: 新機能開発用ブランチ
- `bugfix/*`: バグ修正用ブランチ

### コミットメッセージ

[Conventional Commits](https://www.conventionalcommits.org/) 形式を使用してください：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**タイプ:**
- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメントのみの変更
- `style`: コードの動作に影響しない変更（空白、フォーマットなど）
- `refactor`: バグ修正や機能追加ではないコード変更
- `test`: テストの追加や修正
- `chore`: ビルドプロセスやツールの変更

**例:**
```
feat: add sort functionality for cost analysis results

Add --sort and --sort-field options to enable sorting by input/output/total tokens, cost, or date.
This enhancement improves data analysis capabilities for users.
```

## プルリクエストのガイドライン

1. **フォーク**: リポジトリをフォークし、feature ブランチで作業してください
2. **テスト**: 新しいコードには適切なテストを追加してください
3. **ドキュメント**: 必要に応じてドキュメントを更新してください
4. **コード品質**: 上記のツールでコード品質を確認してください
5. **説明**: プルリクエストには変更内容の明確な説明を含めてください

### プルリクエストテンプレート

```markdown
## 概要
この変更の概要を記述してください。

## 変更内容
- [ ] 新機能の追加
- [ ] バグ修正
- [ ] ドキュメントの更新
- [ ] テストの追加・修正
- [ ] リファクタリング

## テスト
- [ ] 既存のテストがすべて通過することを確認しました
- [ ] 新しい機能/修正に対するテストを追加しました
- [ ] 手動テストを実行しました

## チェックリスト
- [ ] コードがプロジェクトのスタイルガイドラインに従っています
- [ ] 自己レビューを実施しました
- [ ] 必要に応じてドキュメントを更新しました
- [ ] 変更により破壊的な変更は発生しません
```

## 問題の報告

バグや機能要求を報告する際は、以下の情報を含めてください：

### バグ報告

- **環境情報**: OS、Pythonバージョン、パッケージバージョン
- **再現手順**: 問題を再現するための詳細な手順
- **期待される動作**: 何が起こるべきかの説明
- **実際の動作**: 実際に何が起こったかの説明
- **エラーメッセージ**: 関連するエラーメッセージやログ

### 機能要求

- **概要**: 提案する機能の明確な説明
- **動機**: なぜこの機能が必要かの説明
- **使用例**: 機能の使用方法の例
- **代替案**: 検討した他の解決策があれば記述

## ライセンスと著作権

### ライセンス

このプロジェクトは **Apache License 2.0** の下でライセンスされています。貢献することで、あなたの貢献が同じライセンスの下で配布されることに同意したことになります。

### Contributor License Agreement (CLA)

プルリクエストを提出することで、以下に同意したものとみなされます：

1. **権利の付与**: あなたは、あなたの貢献を Apache License 2.0 の条件下で使用、変更、配布する権利をプロジェクトに付与します。

2. **オリジナリティ**: あなたの貢献は、あなた自身のオリジナルの作品であるか、適切にライセンスされた第三者の作品であることを表明します。

3. **特許権**: あなたは、あなたの貢献に関連する特許権があれば、それをプロジェクトに無償でライセンスします。

### 著作権表示

新しいファイルを作成する場合は、以下の著作権表示を含めてください：

```python
# Copyright 2025 Claude Code Cost Collector Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

## リリースプロセス

### バージョニング

このプロジェクトは [Semantic Versioning (SemVer)](https://semver.org/) に従います：

- **MAJOR**: 非互換な API 変更
- **MINOR**: 後方互換性のある機能追加
- **PATCH**: 後方互換性のあるバグ修正

### リリースノート

各リリースには以下を含むリリースノートを作成します：

- **新機能**: 追加された新機能
- **改善**: 既存機能の改善
- **バグ修正**: 修正されたバグ
- **破壊的変更**: 非互換な変更（該当する場合）
- **移行ガイド**: 破壊的変更がある場合の移行方法

## コミュニティとサポート

### コミュニケーション

- **GitHub Issues**: バグ報告や機能要求
- **GitHub Discussions**: 一般的な質問や議論
- **Pull Requests**: コードの貢献

### 行動規範

私たちは、すべての参加者にとって嫌がらせのない体験を提供することにコミットしています。すべての貢献者は以下の原則に従うことが期待されます：

- **尊重**: 異なる視点や経験を尊重する
- **建設的**: 建設的で親切なフィードバックを提供する
- **包括的**: すべてのバックグラウンドの人々を歓迎する
- **プロフェッショナル**: プロフェッショナルで礼儀正しい言動を心がける

## 質問がある場合

このガイドで扱われていない質問がある場合は、GitHub Issues で質問してください。コミュニティメンバーがサポートします。

---

**貢献に感謝します！** あなたの貢献により、Claude Code Cost Collector がより良いツールになります。