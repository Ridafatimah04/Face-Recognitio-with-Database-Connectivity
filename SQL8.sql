-- Create the database if it does not exist
CREATE DATABASE IF NOT EXISTS attendance;

-- Switch to the `attendance` database
USE attendance;

-- Create the `persons` table if it does not already exist
CREATE TABLE IF NOT EXISTS persons (
    id INT AUTO_INCREMENT PRIMARY KEY,          -- Auto-incrementing ID
    name VARCHAR(255) NOT NULL,                 -- Name of the person
    register_number VARCHAR(255) NOT NULL UNIQUE, -- Unique registration number
    attendance TINYINT DEFAULT 0                -- Attendance status (default: 0)
);

-- Display the structure of the `persons` table
DESC persons;

-- Insert or ignore records into the `persons` table
INSERT INTO persons (name, register_number)
VALUES 
    ('rida', 'REG001'),
    ('vaish', 'REG002'),
    ('ayesha', 'REG003'),
    ('sufiya', 'REG004')
ON DUPLICATE KEY UPDATE
    name = VALUES(name);

-- Select all records from the `persons` table
SELECT * FROM persons;

