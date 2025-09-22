import asyncpg
import asyncio
from typing import Any, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.core.config import settings
from langsmith import traceable
 
 
_TRANSIENT_ERRORS = (asyncpg.PostgresError, asyncio.TimeoutError) # Added asyncio.TimeoutError for robustness
 
def with_retry() -> Any:
    """Exponential-back-off (0.2 s → 5 s, 5 tries) for transient errors."""
    return retry(
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        wait=wait_exponential(multiplier=0.2, min=0.2, max=5),
        stop=stop_after_attempt(5),
        reraise=True,
    )
 
# Connection pool for asyncpg
class Database:
    _pool: asyncpg.Pool | None = None
    _init_lock: asyncio.Lock = asyncio.Lock()
    DEFAULT_SCHEMA: str = "public"
 
    @classmethod
    async def init(cls) -> None:
        """Create the global asyncpg pool."""
        if cls._pool is None:
            try:
                print("Initializing database connection pool...")
                cls._pool = await asyncpg.create_pool(
                    host=settings.db_host,
                    port=settings.db_port,
                    database=settings.db_database,
                    user=settings.db_user,
                    password=settings.db_password,
                    ssl="require",
                    timeout=30,
                    min_size=1,
                    max_size=10
                )
                print("Database pool initialized successfully")
            except Exception as e:
                print(f"Error initializing database pool: {str(e)}")
                raise
 
    @classmethod
    async def close(cls) -> None:
        async with cls._init_lock:
            if cls._pool and not cls._pool._closed:
                try:
                    await cls._pool.close()
                    cls._pool = None
                except Exception:
                    raise
            else:
                print("DB pool already closed or not initialized.")
               
    # ── generic helpers ─────────────────────────────────────────────────── #
    @classmethod
    @with_retry()
    async def fetch(cls, query:str, *args) -> list[asyncpg.Record]:
        if cls._pool is None or cls._pool._closed:
            raise RuntimeError("Database pool is not initialized or is closed.")
        async with cls._pool.acquire() as con:
            return await con.fetch(query, *args)
 
    @classmethod
    @with_retry()
    async def fetchrow(cls, query:str, *args) -> asyncpg.Record|None:
        if cls._pool is None or cls._pool._closed:
            raise RuntimeError("Database pool is not initialized or is closed.")
        async with cls._pool.acquire() as con:
            return await con.fetchrow(query, *args)
 
    @classmethod
    @with_retry()
    async def execute(cls, query:str, *args) -> str:
        if cls._pool is None or cls._pool._closed:
            raise RuntimeError("Database pool is not initialized or is closed.")
        async with cls._pool.acquire() as con:
            return await con.execute(query, *args)
   
    @classmethod
    @with_retry()
    async def execute_queries(cls, queries: List[str], *args) -> List[list]:
        """Execute a list of SQL queries in parallel and return the results."""
        try:
            tasks = [cls.execute(query, *args) for query in queries]
            results = await asyncio.gather(*tasks)
            return results
        except RuntimeError:
            raise
        except Exception as e:
            return [f"Error: {e}"]
 
    @classmethod
    @with_retry()
    async def ping(cls) -> bool:
        """Ping the database to check if it is available."""
        if cls._pool is None or cls._pool._closed:
            return False
        try:
            async with cls._pool.acquire(timeout=5) as con:
                result = await con.fetch("SELECT 1;")
                if result:
                    return True
        except Exception:
            return False
        return False
 
   
    @classmethod
    @with_retry()
    async def fetch_batch_vector_search(cls, query_vectors: List[List[float]], limit: int = 5) -> list:
        """Perform a batch vector search for arxiv data based on a list of query vectors."""
        try:
            if cls._pool is None or cls._pool._closed:
                raise RuntimeError("Database pool is not initialized or is closed.")
            
            if not query_vectors:
                return []

            # Convert all vectors to the appropriate string format for SQL compatibility
            query_vectors_str = [
                f"[{','.join(map(str, vector))}]" for vector in query_vectors
            ]

            # Generate a list of tasks for each vector in the batch
            # Each task will acquire its own connection from the pool
            tasks = [
                cls._fetch_vector_for_single_query_with_connection(vector_str, limit)
                for vector_str in query_vectors_str
            ]
            # Gather the results for all the queries
            results = await asyncio.gather(*tasks)

            # Flatten the results into a single list
            return [item for sublist in results for item in sublist]
        
        except Exception as e:
            print(f"Error in fetch_batch_vector_search: {str(e)}")
            return []

    @classmethod
    async def _fetch_vector_for_single_query_with_connection(cls, vector_str: str, limit: int) -> list:
        """Helper function to fetch results for a single vector query with its own connection."""
        if cls._pool is None or cls._pool._closed:
            raise RuntimeError("Database pool is not initialized or is closed.")
            
        async with cls._pool.acquire() as con:
            sql = """
                SELECT title, abstract, category
                FROM arxiv
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1
                LIMIT $2;
            """
            # Fetching results for a single query vector
            results = await con.fetch(sql, vector_str, limit)

            # Return the results in a structured format
            return [
                {
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "category": row["category"]
                }
                for row in results
            ]

    @classmethod
    async def _fetch_vector_for_single_query(cls, con, vector_str: str, limit: int) -> list:
        """Helper function to fetch results for a single vector query."""
        sql = """
            SELECT title, abstract, category
            FROM arxiv
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $2;
        """
        # Fetching results for a single query vector
        results = await con.fetch(sql, vector_str, limit)

        # Return the results in a structured format
        return [
            {
                "title": row["title"],
                "abstract": row["abstract"],
                "category": row["category"]
            }
            for row in results
        ]
 