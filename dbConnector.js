require('dotenv').config();
const mysql = require('mysql2/promise');
const winston = require('winston');

// Setup winston logger
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
    ),
    transports: [
        new winston.transports.File({ filename: 'log.txt' }),
        new winston.transports.Console()
    ],
});

// Create a MySQL connection pool
const pool = mysql.createPool({
    host: process.env.DB_HOST || 'localhost',
    user: process.env.DB_USER || 'root',
    password: process.env.DB_PASSWORD || 'password',
    database: process.env.DB_NAME || 'my_database',
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
});

// Function to execute a query with error handling and logging
async function executeQuery(sql, params) {
    try {
        logger.info(`Executing query: ${sql}`);
        const [results] = await pool.execute(sql, params);
        return results;
    } catch (err) {
        logger.error(`Query failed: ${err.message}`);
        throw err;
    }
}

// Function to close the pool gracefully
async function closeConnection() {
    try {
        await pool.end();
        logger.info('Database connection closed.');
    } catch (err) {
        logger.error(`Error closing connection: ${err.message}`);
        throw err;
    }
}

// Example function for inserting data
async function insertData(table, data) {
    const keys = Object.keys(data).join(', ');
    const placeholders = Object.keys(data).map(() => '?').join(', ');
    const values = Object.values(data);

    const sql = `INSERT INTO ${table} (${keys}) VALUES (${placeholders})`;
    return await executeQuery(sql, values);
}

// Example function for retrieving data
async function fetchData(table, criteria = {}) {
    const keys = Object.keys(criteria).map(key => `${key} = ?`).join(' AND ');
    const values = Object.values(criteria);

    const sql = `SELECT * FROM ${table}` + (keys ? ` WHERE ${keys}` : '');
    return await executeQuery(sql, values);
}

module.exports = {
    executeQuery,
    insertData,
    fetchData,
    closeConnection,
};
