require('dotenv').config();
const mongoose = require('mongoose');

// MongoDB connection
const MONGODB_URI = 'mongodb+srv://yashshah25112003:uQDoppAs6b74ipTD@ta-ra-assistant.djncl4r.mongodb.net/?retryWrites=true&w=majority&appName=TA-RA-ASSISTANT';

mongoose.connect(MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
.then(() => console.log('Connected to MongoDB'))
.catch(err => console.error('MongoDB connection error:', err));

// Export the mongoose connection
module.exports = mongoose.connection; 