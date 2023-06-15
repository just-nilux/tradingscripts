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
                                        ask_price text,
                                        fill_price text,
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
                                        STOP_LIMIT_ID text,
                                        ask_TP text,
                                        ask_SL text,
                                        fill_TP text,
                                        fill_SL text,
                                        Pnl text
                                    ); """)
        except Error as e:
            self.logger.error(f"{e}")



    def insert_position(self, position_data, TAKE_PROFIT, STOP_LIMIT, entrystrat, timeframe):
        """ insert a new position into the positions table """
        sql = ''' INSERT INTO positions(id, market, side, ask_price, fill_price, triggerPrice, size, type, createdAt, unfillableAt, expiresAt, status, cancelReason, entryStrat,
                timeframe, TAKE_PROFIT_ID, STOP_LIMIT_ID, ask_TP, ask_SL, fill_TP, fill_SL, Pnl) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''

        try:
            created_at = datetime.strptime(position_data['createdAt'], "%Y-%m-%dT%H:%M:%S.%fZ") if position_data['createdAt'] else None
            expires_at = datetime.strptime(position_data['expiresAt'], "%Y-%m-%dT%H:%M:%S.%fZ") if position_data['expiresAt'] else None
        except ValueError as e:
            self.logger.error(f"Error parsing date: {e}")
            return

        #try:
            #ask_price = float(position_data['price']) if position_data['price'] else None
            #size = float(position_data['size']) if position_data['size'] else None

            #TP = float(TAKE_PROFIT['price']) if TAKE_PROFIT['price'] else None
            #SL = float(STOP_LIMIT['triggerPrice']) if STOP_LIMIT['triggerPrice'] else None
        #except ValueError as e:
            #self.logger.error(f"Error converting price or size to float: {e}")
            #return

        position_tuple = (
            position_data['id'],
            position_data['market'],
            position_data['side'],
            position_data['price'],
            None,
            position_data['triggerPrice'],
            position_data['size'],
            position_data['type'],
            created_at,
            position_data['unfillableAt'],
            expires_at,
            position_data['status'],
            position_data['cancelReason'],
            entrystrat,
            timeframe,
            TAKE_PROFIT['id'],
            STOP_LIMIT['id'],
            TAKE_PROFIT['price'],
            STOP_LIMIT['triggerPrice'],
            None,
            None,
            None  # Pnl initialized as None
        )
        try:
            with self.conn:
                cur = self.conn.cursor()
                cur.execute(sql, position_tuple)
        except Error as e:
            self.logger.error(f"Error inserting position: {e}")

    


def update_position_status(self, client):
    """ Update status of each open position in the positions table """

    # SQL statement to select open positions
    select_sql = "SELECT id, TAKE_PROFIT_ID, STOP_LIMIT_ID FROM positions WHERE status = 'PENDING'"

    # SQL statement to update position data
    update_sql = '''UPDATE positions SET status = ?, unfillableAt = ?, fill_price = ?,  fill_TP = ?, fill_SL = ? WHERE id = ?'''

    try:
        with self.conn:
            cur = self.conn.cursor()

            # Execute SELECT statement
            cur.execute(select_sql)
            open_positions = cur.fetchall()

            for pos_id, tp_id, sl_id in open_positions:
                # Call the dydx exchange API for each open position
                response = client.private.get_order_by_id(pos_id).data['order']

                fills = client.private.get_fills(order_id=pos_id).data['fills']
                fill_price = fills[0].get('price') if fills else None

                tp_fills = client.private.get_fills(order_id=tp_id).data['fills']
                fill_TP = tp_fills[0].get('price') if tp_fills else None

                sl_fills = client.private.get_fills(order_id=sl_id).data['fills']
                fill_SL = sl_fills[0].get('price') if sl_fills else None

                # Extract data from API response
                status = response['status']
                unfillable_at = datetime.strptime(response['unfillableAt'], "%Y-%m-%dT%H:%M:%S.%fZ") if response['unfillableAt'] else None

                # Execute UPDATE statement
                cur.execute(update_sql, (status, unfillable_at, fill_price, fill_TP, fill_SL, pos_id))
    except Error as e:
        self.logger.error(f"Error updating position status: {e}")




    def get_fields_by_id(self, fields, id):
        """ Retrieve specific fields from a record by its id """

        fields_str = ", ".join(fields)
        sql = f'SELECT {fields_str} FROM positions WHERE id = ?'

        try:
            with self.conn:
                cur = self.conn.cursor()
                cur.execute(sql, (id,))
                result = cur.fetchone()
                if result:
                    return result
        except Error as e:
            self.logger.error(f"Error retrieving fields: {e}")



    def get_order_ids_for_open_positions(self):
        """ Retrieve TAKE_PROFIT_ID and STOP_LIMIT_ID from records where status is NOT 'CLOSED' """

        sql = 'SELECT TAKE_PROFIT_ID, STOP_LIMIT_ID FROM positions WHERE status <> "CLOSED"'
        try:
            with self.conn:
                cur = self.conn.cursor()
                cur.execute(sql)
                results = cur.fetchall()
                if results:
                    return results
        except Error as e:
            self.logger.error(f"Error retrieving fields: {e}")



    def get_record_by_order_id(self, order_id):
        """ Retrieve a record by TAKE_PROFIT_ID or STOP_LIMIT_ID """
        try:
            with self.conn:
                cur = self.conn.cursor()
                cur.execute('SELECT id, TAKE_PROFIT_ID, STOP_LIMIT_ID FROM positions WHERE TAKE_PROFIT_ID = ? OR STOP_LIMIT_ID = ?', (order_id, order_id))
                result = cur.fetchone()
                if result:
                    # If the order_id provided matches TAKE_PROFIT_ID, return the ID and STOP_LIMIT_ID.
                    if result[1] == order_id:
                        return result[0], result[2]
                    # If the order_id provided matches STOP_LIMIT_ID, return the ID and TAKE_PROFIT_ID.
                    else:
                        return result[0], result[1]
        except Error as e:
            self.logger.error(f"Error retrieving record: {e}")



    def update_status_by_id(self, id, new_status):
        """ Update status of a record by id """
        try:
            with self.conn:
                cur = self.conn.cursor()
                cur.execute('UPDATE positions SET status = ? WHERE id = ?', (new_status, id))
                self.conn.commit()  # save the changes
        except Error as e:
            self.logger.error(f"Error updating status: {e}")







        



