import psycopg2
import os

def init_db():
    # docker-compose.yml で設定した環境変数から接続情報を取得
    db_url = os.environ.get('DATABASE_URL')
    conn = psycopg2.connect(db_url)
    
    with conn.cursor() as cur:
        # 1. プロジェクト単位
        cur.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # 2. 実験セット単位（使用モデルなど）
        cur.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id SERIAL PRIMARY KEY,
                project_id INTEGER REFERENCES projects(id),
                provider TEXT,
                model_name TEXT,
                n_range TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # 3. 個別の試行データ（Accuracyやトークン数）
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                id SERIAL PRIMARY KEY,
                experiment_id INTEGER REFERENCES experiments(id),
                n INTEGER,
                accuracy FLOAT,
                token_count INTEGER,
                final_score FLOAT
            );
        """)
        print("テーブルの作成に成功しました！")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()