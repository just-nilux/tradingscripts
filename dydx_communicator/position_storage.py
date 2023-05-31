from logger_setup import setup_logger
from sqlite3 import Error
import sqlite3



class PositionStorage:
    def __init__(self, db_file):
        self.logger = setup_logger(__name__)

        """ create a database connection to a SQLite database """
        self.conn = None
        try:
            self.conn = sqlite3.connect(db_file)
        except Error as e:
            self.logger.error(f"{e}")            
        if self.conn:
            self.create_table()



    def create_table(self):
        """ create a table for the positions """
        try:
            cur = self.conn.cursor()
            cur.execute(""" CREATE TABLE IF NOT EXISTS positions (
                                        market text,
                                        createdAt text,
                                        entryStrat text,
                                        timeframe text
                                    ); """)
        except Error as e:
            print(e)
            self.logger.error(f"{e}")


    
    def insert_position(self, position_data, entry_strat, timeframe):
        """ insert a new position into the positions table """
        sql = ''' INSERT INTO positions(market, createdAt, entryStrat)
                  VALUES(?,?,?,?) '''
        position_tuple = (
            position_data['market'],
            position_data['createdAt'],
            entry_strat,
            timeframe
        )
        try:
            cur = self.conn.cursor()
            cur.execute(sql, position_tuple)
            self.conn.commit()
            return cur.lastrowid
        except Error as e:
            self.logger.error(f"{e}")     



    def close(self):
        """ close the database connection """
        if self.conn:
            self.conn.close()
