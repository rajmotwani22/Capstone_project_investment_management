import sqlite3

DATABASE = "financial_data.db"

def view_data():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_data")
        rows = cursor.fetchall()
        for row in rows:
            print(row)

view_data()
