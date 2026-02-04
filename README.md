# ELEC490 Capstone Project - AI for Fake News Detection

# 1. Rebuild the image (this takes a minute or two)
docker-compose build --no-cache backend

# 2. Restart the container
docker-compose up -d

# 3. ONLY IF: you add new data into the db and need to run embeddings
docker-compose exec backend uv run python src/vector_db/run_load_embeddings.py
