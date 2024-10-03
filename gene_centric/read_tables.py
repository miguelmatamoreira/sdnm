import pandas as pd
from sqlalchemy import create_engine, text



def get_column_data_types(df):
    '''
    Checks type of data to create table.

    Parameters:
    - df (dataset): Dataset to be read.

    Returns:
    - data_types (str): String containing the data types.
    '''
    data_types = []
    for column_name, dtype in df.dtypes.items():
        if dtype == 'object':
            data_types.append(f'"{column_name}" VARCHAR')
        elif 'int' in str(dtype):
            data_types.append(f'"{column_name}" INTEGER')
        elif 'float' in str(dtype):
            data_types.append(f'"{column_name}" FLOAT')
        elif 'bool' in str(dtype):
            data_types.append(f'"{column_name}" BOOLEAN')
        else:
            data_types.append(f'"{column_name}" VARCHAR') 
    return ', '.join(data_types)



def create_table(table_name, csv_file_path):
    '''
    Create a table in the database.

    Parameters:
    - engine (engine): Database informations.
    - table_name (str): Name of the table to be create.
    - csv_file_path (str): Path of the csv that contains the data to the table.
    '''
    engine = create_engine('postgresql://ist195643:2800@db.tecnico.ulisboa.pt:5432/ist195643')
    df = pd.read_csv(csv_file_path, sep='\t', skiprows=2, index_col=0)
    column_data_types = get_column_data_types(df)
    # create the query to create table
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        {column_data_types}
    );
    """
    with engine.connect() as connection:
        connection.execute(text(create_table_query))
    # add all the data in chunks
    chunk_size = 1000
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        chunk.to_sql(table_name, con=engine, if_exists='append', index=False)
        print(f"Chunk {i} inserted!")
    print("Data inserted successfully!")
    print("Table created and data inserted successfully!")
    engine.dispose()



def delete_table(engine, table_name):
    '''
    Delete a table in the database.
    
    Parameters:
    - engine (engine): Database informations.
    - table_name (str): Name of the table to be create.
    '''
    with engine.connect() as connection:
        query = f"DROP TABLE IF EXISTS {table_name};"
        connection.execute(query)
    print(f"Table '{table_name}' deleted successfully!")



def read_values(table_name, genes):
    '''
    Reads the genes from the table.

    Parameters:
    - table_name (str): Table to look at in the database.
    - genes (list of str): List of genes to retrieve from the table.

    Returns:
    - df_from_db (dataset): Part of the table with genes of interest.
    '''
    engine = create_engine('postgresql://ist195643:2800@db.tecnico.ulisboa.pt:5432/ist195643')
    select_query = f'SELECT * FROM {table_name} WHERE "Description" IN {tuple(genes)}'
    df_from_db = pd.read_sql(select_query, con=engine)
    # print(df_from_db.head())
    # print(df_from_db.shape)
    # print(f"Read {table_name} successfully!")
    engine.dispose()
    return df_from_db



if __name__ == "__main__":
    print("Run the app.py")