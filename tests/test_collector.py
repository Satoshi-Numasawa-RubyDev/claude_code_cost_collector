"""collector.pyのユニットテスト"""

import tempfile
import unittest
from pathlib import Path

from claude_code_cost_collector.collector import collect_log_files


class TestCollectLogFiles(unittest.TestCase):
    """collect_log_files関数のテストクラス"""

    def setUp(self):
        """テストデータの準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # テスト用JSONファイルを作成
        self.json1 = self.temp_path / "log1.json"
        self.json2 = self.temp_path / "subdir" / "log2.json"
        self.txt_file = self.temp_path / "not_json.txt"

        # ディレクトリ作成
        (self.temp_path / "subdir").mkdir(exist_ok=True)

        # ファイル作成
        self.json1.write_text('{"test": "data1"}')
        self.json2.write_text('{"test": "data2"}')
        self.txt_file.write_text("not json content")

    def tearDown(self):
        """テストデータのクリーンアップ"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_collect_log_files_success(self):
        """正常ケース: JSONファイルが正しく収集されること"""
        result = collect_log_files(self.temp_dir)

        # JSONファイルのみが収集されることを確認
        self.assertEqual(len(result), 2)

        # 結果にjson1とjson2が含まれることを確認
        result_names = {p.name for p in result}
        self.assertIn("log1.json", result_names)
        self.assertIn("log2.json", result_names)

        # txtファイルは含まれないことを確認
        self.assertNotIn("not_json.txt", result_names)

        # 全てPathオブジェクトであることを確認
        self.assertTrue(all(isinstance(p, Path) for p in result))

    def test_collect_log_files_directory_not_found(self):
        """存在しないディレクトリの場合、FileNotFoundErrorが発生すること"""
        non_existent_dir = self.temp_path / "non_existent"

        with self.assertRaises(FileNotFoundError) as context:
            collect_log_files(non_existent_dir)

        self.assertIn("Directory not found", str(context.exception))


if __name__ == "__main__":
    unittest.main()
