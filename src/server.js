const https = require("https");
const fs = require('fs');
const cors = require("cors");
const express = require("express");

const PORT = process.env.PORT || 8765;


const app = express();
app.use(cors({
    origin: "*",
}));
const server = https.createServer({
    cert: fs.readFileSync('./.cert/cert.pem'),
    key: fs.readFileSync('./.cert/key.pem'),
},  app);


app.use("/senddata", (req, res) => {
    req.pipe(fs.createWriteStream("tmp/file.png"));    
    console.log("Image recieved");
    res.json({ success: true });
})

server.listen(PORT, () => {
    console.log(`Server is listening on port ${PORT}`);
});

