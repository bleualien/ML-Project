from app import db, app  # import your Flask app and db
from sqlalchemy import text  # <-- important

with app.app_context():
    # 1️⃣ Create all tables (if not exists)
    db.create_all()
    print("All tables created successfully!")

    # 2️⃣ Check if 'department' column exists
    table_name = 'detection_department'
    column_name = 'department'

    query = text(f"""
    SELECT column_name 
    FROM information_schema.columns
    WHERE table_name='{table_name}' AND column_name='{column_name}';
    """)

    # Use a connection
    with db.engine.connect() as conn:
        result = conn.execute(query)
        if result.fetchone():
            print(f"Column '{column_name}' already exists in '{table_name}'.")
        else:
            # 3️⃣ Add column safely (nullable first)
            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} VARCHAR;"))
            print(f"Column '{column_name}' added successfully!")

            # Optional: Fill existing rows with default value
            conn.execute(text(f"UPDATE {table_name} SET {column_name}='Ward Office' WHERE {column_name} IS NULL;"))
            print(f"Existing rows updated with default 'Ward Office'.")

            # Optional: Make column NOT NULL
            conn.execute(text(f"ALTER TABLE {table_name} ALTER COLUMN {column_name} SET NOT NULL;"))
            print(f"Column '{column_name}' set to NOT NULL.")
