var whois = require('whois')
var fs = require('fs');
var async = require('async');
var obj = JSON.parse(fs.readFileSync('ips.json', 'utf8'));



function lookup(IP, callback) {
    whois.lookup(IP, function(err, data) {
        try {
            // Get whatever info you want out of the whoIs here
            var result = data.split(/Organization:   /)[1].split("RegDate")[0].trim()
            callback(result)}
        catch (ex) {
            callback("Null")
        }
    });
    
}

dict = {}

var bigPromises = Object.keys(obj).map(function(key){
    return new Promise(function(resolve, reject) {
        var ips = obj[key];

        var promises = ips.map(function(ip){
            return new Promise(function(resolve, reject) {
                var res = lookup(ip, function(result) {
                    resolve(result)
                });
            });
        });

        Promise.all(promises).then(function(value) {
            resolve([key, value])
        }, function(err) {console.log(key);});
    })
})

Promise.all(bigPromises).then(function(vals){
    var dict = {}
    vals.forEach(function(pair) {
        dict[pair[0]] = pair[1]
    });
    console.log(dict);
    fs.writeFileSync("companies.json", JSON.stringify(dict, null, '\t'));
})