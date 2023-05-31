import sqlite3
from sqlite3 import Error

class PositionStorage:
    def __init__(self, db_file):
        """ create a database connection to a SQLite database """
        self.conn = None
        try:
            self.conn = sqlite3.connect(db_file)
        except Error as e:
            print(e)
            
        if self.conn:
            self.create_table()

    def create_table(self):
        """ create a table for the positions """
        try:
            cur = self.conn.cursor()
            cur.execute(""" CREATE TABLE IF NOT EXISTS positions (
                                        market text,
                                        createdAt text,
                                        entryStrat text
                                    ); """)
        except Error as e:
            print(e)


    
    def insert_position(self, position_data, entry_strat):
        """ insert a new position into the positions table """
        sql = ''' INSERT INTO positions(market, createdAt, entryStrat)
                  VALUES(?,?,?) '''
        position_tuple = (
            position_data['market'],
            position_data['createdAt'],
            entry_strat
        )
        cur = self.conn.cursor()
        cur.execute(sql, position_tuple)
        self.conn.commit()
        return cur.lastrowid



    def close(self):
        """ close the database connection """
        if self.conn:
            self.conn.close()
