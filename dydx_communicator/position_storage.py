from logger_setup import setup_logger
from sqlite3 import Error
import sqlite3
from datetime import datetime

class PositionStorage:
    def __init__(self, db_file):
        self.logger = setup_logger(__name__)

        """ create a database connection to a SQLite database """
        self.conn = None
        try:
            self.conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
        except Error as e:
            self.logger.error(f"{e}")            
        if self.conn:
            self.create_table()

    def create_table(self):
        """ create a table for the positions """
        try:
            cur = self.conn.cursor()
            cur.execute(""" CREATE TABLE IF NOT EXISTS positions (
                                        id text,
                                        market text,
                                        side text,
                                        price real,
                                        triggerPrice text,
                                        size real,
                                        type text,
                                        createdAt timestamp,
                                        unfillableAt text,
                                        expiresAt timestamp,
                                        status text,
                                        cancelReason text,
                                        entryStrat text,
                                        timeframe text,
                                        TAKE_PROFIT_ID text,
                                        STOP_LIMIT_ID text
                                    ); """)
        except Error as e:
            self.logger.error(f"{e}")



    def insert_position(self, position_data, TAKE_PROFIT, STOP_LIMIT, entrystrat, timeframe):
        """ insert a new position into the positions table """
        sql = ''' INSERT INTO positions(id, market, side, price, triggerPrice, size, type, createdAt, unfillableAt, expiresAt, status, cancelReason, entryStrat,
                timeframe, TAKE_PROFIT_ID, STOP_LIMIT_ID) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''

        try:
            created_at = datetime.strptime(position_data['createdAt'], "%Y-%m-%dT%H:%M:%S.%fZ") if position_data['createdAt'] else None
            expires_at = datetime.strptime(position_data['expiresAt'], "%Y-%m-%dT%H:%M:%S.%fZ") if position_data['expiresAt'] else None
        except ValueError as e:
            self.logger.error(f"Error parsing date: {e}")
            return

        try:
            price = float(position_data['price']) if position_data['price'] else None
            size = float(position_data['size']) if position_data['size'] else None
        except ValueError as e:
            self.logger.error(f"Error converting price or size to float: {e}")
            return

        position_tuple = (
            position_data['id'],
            position_data['market'],
            position_data['side'],
            price,
            position_data['triggerPrice'],
            size,
            position_data['type'],
            created_at,
            position_data['unfillableAt'],
            expires_at,
            position_data['status'],
            position_data['cancelReason'],
            entrystrat,
            timeframe,
            TAKE_PROFIT,
            STOP_LIMIT
        )
        try:
            with self.conn:
                cur = self.conn.cursor()
                cur.execute(sql, position_tuple)
        except Error as e:
            self.logger.error(f"Error inserting position: {e}")


