import os
import json
import psycopg2
from psycopg2.extras import execute_values

def migrate():
    db_url = os.environ.get('DATABASE_URL')
    conn = psycopg2.connect(db_url)
    results_dir = 'experiment_results' # インポート対象のフォルダ

    with conn.cursor() as cur:
        # プロジェクトIDを取得（なければ作成）
        cur.execute("INSERT INTO projects (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name RETURNING id", ("Hanoi Collapse Research",))
        project_id = cur.fetchone()[0]

        # フォルダ内の全JSONをスキャン
        files = [f for f in os.listdir(results_dir) if f.startswith('results_N') and f.endswith('.json')]
        print(f"{len(files)} 件のファイルをインポート開始...")

        for filename in files:
            with open(os.path.join(results_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 実験メタデータの登録（まだなければ）
            cur.execute("""
                INSERT INTO experiments (project_id, provider, model_name, n_range)
                VALUES (%s, %s, %s, %s) RETURNING id
            """, (project_id, data['provider'], data.get('model', 'unknown'), f"{data['N']}"))
            exp_id = cur.fetchone()[0]

            # 試行データの登録
            cur.execute("""
                INSERT INTO trials (experiment_id, n, accuracy, token_count, final_score)
                VALUES (%s, %s, %s, %s, %s)
            """, (exp_id, data['N'], data['accuracy'], data['token_count'], data['final_score']))
            
        conn.commit()
        print("インポートが完了しました！")

    conn.close()

if __name__ == "__main__":
    migrate()