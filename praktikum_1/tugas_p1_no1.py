# Import library
import mysql.connector
from mysql.connector import errorcode
from pandas import DataFrame, ExcelWriter

# Create dictionary for database configuration
config = {
    'user': 'root',
    'password': '',      # tambahkan password jika ada
    'host': 'localhost',
    'database': 'pm_24sa11a159'
}

# Create class
class DatabaseConnection:

    # Constructor
    def __init__(self):
        try:
            # Create database connection
            self.db_connect = mysql.connector.connect(**config)

            # Create cursor to execute SQL query
            self.cursor = self.db_connect.cursor()

        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Username atau password salah")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database tidak ditemukan")
            else:
                print(err)

    # Function to execute SELECT query
    def Select(self, query):
        self.cursor.execute(query)
        data = self.cursor.fetchall()
        return data

    # Destructor
    def __del__(self):
        if hasattr(self, 'cursor'):
            self.cursor.close()

        if hasattr(self, 'db_connect'):
            self.db_connect.close()


# ==========================================
# Get data from MySQL
db = DatabaseConnection()

query = """
SELECT m.nim, m.nama_mahasiswa, p.nama_prodi, d.nama_dosen
FROM mahasiswa AS m INNER JOIN prodi AS p ON m.id_prodi = p.id_prodi
INNER JOIN dosen_pa AS d ON m.id_dosen_pa = d.id_dosen_pa
"""

data = db.Select(query)
print(data)

# ==========================================
# Output to Excel

dataset = DataFrame(
    data,
    columns=['nim', 'nama_mahasiswa', 'nama_prodi', 'nama_dosen']
)

writer = ExcelWriter('data_siswa_igo-tegar-prambudhy.xlsx')
dataset.to_excel(writer, sheet_name='Mahasiswa', index=False)

writer.close()