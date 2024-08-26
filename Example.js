const { insertData, fetchData } = require('./dbConnector');

async function saveCommand(command, response) {
    await insertData('commands', { command, response });
}

async function getCommandHistory() {
    return await fetchData('commands');
}

// Example of saving a command and response
saveCommand('Analyze data', 'Analysis complete').then(() => {
    console.log('Command saved!');
}).catch(err => console.error(err));

// Example of fetching command history
getCommandHistory().then(history => {
    console.log('Command history:', history);
}).catch(err => console.error(err));
