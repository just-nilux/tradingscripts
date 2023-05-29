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
                                        status text,
                                        side text,
                                        size text,
                                        maxSize text,
                                        entryPrice text,
                                        exitPrice text,
                                        unrealizedPnl text,
                                        realizedPnl text,
                                        createdAt text,
                                        closedAt text,
                                        sumOpen text,
                                        sumClose text,
                                        netFunding text,
                                        entryStrat text
                                    ); """)
        except Error as e:
            print(e)



    def insert_position(self, position):
        """ insert a new position into the positions table """
        sql = ''' INSERT INTO positions(market,status,side,size,maxSize,entryPrice,exitPrice,unrealizedPnl,realizedPnl,createdAt,closedAt,sumOpen,sumClose,netFunding)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''
        cur = self.conn.cursor()
        cur.execute(sql, position)
        return cur.lastrowid



    def close(self):
        """ close the database connection """
        if self.conn:
            self.conn.close()
