import logging
from neo4j import GraphDatabase
from typing import List, Dict, Optional

# Налаштовуємо логер
logger = logging.getLogger("dispatcher.neo4j")


class Neo4jRAG:
    def __init__(self):
        # Можна брати з env, поки хардкод для сумісності з твоїм setup
        uri = "bolt://neo4j:7687"
        auth = ("neo4j", "password123")
        try:
            self.driver = GraphDatabase.driver(uri, auth=auth)
            # Перевірка з'єднання (легкий пінг)
            self.driver.verify_connectivity()
        except Exception as e:
            logger.error(f"❌ Neo4j Connection Failed: {e}")
            self.driver = None

    def retrieve(
            self,
            epitaph_id: str,
            keywords: List[str],
            dataset_run_id: Optional[str] = None,
            limit: int = 5
    ) -> List[Dict]:
        if not self.driver:
            logger.error("🚫 Neo4j driver is not initialized. Skipping RAG.")
            return []

        # Логуємо вхідні дані
        logger.info(f"🔍 Neo4j Query | Target: '{epitaph_id}' | Keywords: {keywords} | RunID: {dataset_run_id}")

        query = """
        MATCH (m:Message)
        WHERE m.epitaph_id = $epitaph_id
        """

        if dataset_run_id:
            query += " AND m.dataset_run_id = $dataset_run_id "

        query += """
        AND any(k IN $keywords WHERE
          toLower(m.text) CONTAINS toLower(k) OR
          toLower(m.topic) CONTAINS toLower(k)
        )
        RETURN m.text AS text, m.topic AS topic, m.role AS role, m.timestamp AS timestamp, m.seq AS seq
        ORDER BY m.seq DESC
        LIMIT $limit
        """

        try:
            with self.driver.session() as session:
                res = session.run(
                    query,
                    epitaph_id=epitaph_id,
                    dataset_run_id=dataset_run_id,
                    keywords=keywords,
                    limit=limit,
                )

                data = [r.data() for r in res]

                # Логуємо результат
                if data:
                    logger.info(f"✅ Found {len(data)} facts for '{epitaph_id}'")
                    # Прінтуємо перші 50 символів першого факту для перевірки
                    logger.info(f"   Sample: {str(data[0])[:100]}...")
                else:
                    logger.warning(f"⚠️ No facts found for '{epitaph_id}' with keywords {keywords}")

                return data

        except Exception as e:
            logger.error(f"💥 Neo4j Execution Error: {e}")
            return []

    def close(self):
        if self.driver:
            self.driver.close()