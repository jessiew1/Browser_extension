const express = require('express');
const router = express.Router();
const fs = require('fs');

const MongoClient = require('mongodb').MongoClient

const write_to_file = (file_name, json_data) => {
  fs.writeFile(file_name, json_data, function(err) {
    if (err) {
      console.log(err);
    }
  });
}

const read_from_file = (file_name) => {
  return fs.readFileSync(file_name);
}

const save_object_to_mongo = (to_store) => {
  MongoClient.connect('mongodb://localhost:27017/', function (err, client) {
    if (err) {
      throw err
    }

    const db = client.db('MPCVoting')
    const collection = db.collection('shares')

    collection.insertOne(to_store, function (err, result) {
      if (err) {
        throw err
      }
      
      if (result.result.n != 1) {
        throw 'Writing to MongoDB did not seem to go well. result.result.n != 1'
      }
      
      if (result.result.ok != 1) {
        throw 'Writing to MongoDB did not seem to go well. result.result.ok != 1'
      }
      
      console.log('The object was saved to MongoDB successfully!')
      // console.log(result.result)
    })
  })
}

// Ping pong!
router.get('/ping', function(req, res, next) {
  res.send({
    info: 'pong!',
  });
});

// Server sends the public key from a fresh random keypair to the client
// This is just to make sure that the client can properly parse a public key.
router.get('/test1/getFreshRandomPublicKey', function(req, res, next) {
  const _sodium = require('libsodium-wrappers');
  (async() => {
    await _sodium.ready;
    const sodium = _sodium;
    
    const keypair = sodium.crypto_box_keypair();
    const public_key = keypair.publicKey;
    const private_key = keypair.privateKey;
    
    console.log(`${keypair.keyType}`);
    console.log(`${JSON.stringify(public_key)}`);
    
    res.send({
      info: 'No info',
      public_key: public_key,
    })
  })();
});

// Server sends a public key and message to the client.
// The client should encrypt using crypto_box_seal and send
// a POST request to the server at /test2/receiveCiphertext.
router.get('/test2/getPublicKeyAndMessage', function(req, res, next) {
  const _sodium = require('libsodium-wrappers');
  (async() => {
    await _sodium.ready;
    const sodium = _sodium;
    
    const seed = Uint8Array.from(Array.from({length: sodium.crypto_box_SEEDBYTES}, (v, i) => i))
    
    console.log(`${seed}`);
    const keypair = sodium.crypto_box_seed_keypair(seed);
    const public_key = keypair.publicKey;
    
    console.log(`${keypair.keyType}`);
    console.log(`${JSON.stringify(public_key)}`);
    
    // Select message
    const m = 'Hello test2'
    const mb = sodium.from_string(m)
    
    res.send({
      info: 'No info',
      public_key: public_key,
      message: message,
    })
  })();
});

// Server sends the public key from a fresh random keypair to the client
// Query parameter should be named ciphertext and should be of the type Uint8Array
router.post('/test2/receiveCiphertext', function(req, res, next) {
  const _sodium = require('libsodium-wrappers');
  (async() => {
    await _sodium.ready;
    const sodium = _sodium;
    
    const seed = Uint8Array.from(Array.from({length: sodium.crypto_box_SEEDBYTES}, (v, i) => i))
    
    console.log(`${seed}`);
    const keypair = sodium.crypto_box_seed_keypair(seed);
    const public_key = keypair.publicKey;
    const private_key = keypair.privateKey;
    
    console.log(`${keypair.keyType}`);
    console.log(`${JSON.stringify(public_key)}`);
    console.log(`${JSON.stringify(private_key)}`);
    
    // Select message
    const m = 'Hello test2'
    const mb = sodium.from_string(m)
    
    // Encrypt
    const c = req.query.ciphertext;
    
    // Decrypt
    const rb = sodium.crypto_box_seal_open(c, public_key, private_key)
    const r = sodium.to_string(rb)
    
    // Assert correctness
    if (m != r) {
      throw 'm != r'
    }
  })();
  
  res.send({
    info: 'Test passed!',
  })
});

// Make sure to run this at least once before any other function in the test.
// It will use hardcoded seeds to generate 3 keypairs, one for each compute party.
// Then, it will write those keypairs in JSON format to individual files.
router.get('/test3/generateKeypairs', function(req, res, next) {
  const _sodium = require('libsodium-wrappers');
  (async() => {
    await _sodium.ready;
    const sodium = _sodium;
    
    const seed1 = Uint8Array.from(Array.from({length: sodium.crypto_box_SEEDBYTES}, (v, i) => 3 * i))
    const seed2 = Uint8Array.from(Array.from({length: sodium.crypto_box_SEEDBYTES}, (v, i) => 3 * i + 1))
    const seed3 = Uint8Array.from(Array.from({length: sodium.crypto_box_SEEDBYTES}, (v, i) => 3 * i + 2))
    
    const keypair1 = sodium.crypto_box_seed_keypair(seed1);
    const keypair2 = sodium.crypto_box_seed_keypair(seed2);
    const keypair3 = sodium.crypto_box_seed_keypair(seed3);
    
    const public_key1 = keypair1.publicKey;
    const public_key2 = keypair2.publicKey;
    const public_key3 = keypair3.publicKey;
    
    write_to_file('keypair1.json', JSON.stringify(keypair1));
    write_to_file('keypair2.json', JSON.stringify(keypair2));
    write_to_file('keypair3.json', JSON.stringify(keypair3));
    
    res.send({
      info: 'Success.',
    })
  })();
});

// Responds with the public keys from the files.
// This simulates asking for the public keys of the three compute parties..
router.get('/test3/getPublicKeys', function(req, res, next) {
  const json1 = read_from_file('keypair1.json');
  const json2 = read_from_file('keypair2.json');
  const json3 = read_from_file('keypair3.json');
  
  const keypair1 = JSON.parse(json1);
  const keypair2 = JSON.parse(json2);
  const keypair3 = JSON.parse(json3);
  
  const public_key1 = keypair1.publicKey;
  const public_key2 = keypair2.publicKey;
  const public_key3 = keypair3.publicKey;

  res.send({
    info: 'Success.',
    public_key1: public_key1,
    public_key2: public_key2,
    public_key3: public_key3,
  })
});

// Submits ciphertexts to the routing server.
// This simulates sending encrypted data to the routing server.
router.post('/test3/receiveCiphertexts', function(req, res, next) {
  const _sodium = require('libsodium-wrappers');
  (async() => {
    await _sodium.ready;
    const sodium = _sodium;

    console.log(JSON.stringify(req.query));
    
    const c1 = req.query.ciphertext1;
    const c2 = req.query.ciphertext2;
    const c3 = req.query.ciphertext3;
    
    // const to_store = {
      // share_one: c1,
      // share_two: c2,
      // share_three: c3,
    // }
    // save_object_to_mongo(to_store)

    const json1 = read_from_file('keypair1.json');
    const json2 = read_from_file('keypair2.json');
    const json3 = read_from_file('keypair3.json');
    
    const keypair1 = JSON.parse(json1);
    const keypair2 = JSON.parse(json2);
    const keypair3 = JSON.parse(json3);

    const rb1 = sodium.crypto_box_seal_open(c1, keypair1.publicKey, keypair1.privateKey)
    const rb2 = sodium.crypto_box_seal_open(c2, keypair2.publicKey, keypair2.privateKey)
    const rb3 = sodium.crypto_box_seal_open(c3, keypair3.publicKey, keypair3.privateKey)
  
    res.send({
      info: 'These are the messages received (after decrypting). We currently do not test if these messages represent valid data from a user.',
      received1: rb1,
      received2: rb2,
      received3: rb3,
    })
  })();
});

// Submits ciphertexts to the routing server.
// Do this at least once to populate the database so Nathan has some data to test other parts of the system.
router.post('/test3/saveCiphertextsToDatabase', function(req, res, next) {
  const _sodium = require('libsodium-wrappers');
  (async() => {
    await _sodium.ready;
    const sodium = _sodium;
    
    const c1 = req.query.ciphertext1;
    const c2 = req.query.ciphertext2;
    const c3 = req.query.ciphertext3;
    
    const to_store = {
      share_one: c1,
      share_two: c2,
      share_three: c3,
    }
    save_object_to_mongo(to_store)

    res.send({
      info: 'Success.',
    })
  })();
});

// Use the query parameters share_one, share_two, and share_three,
// to make a javascript object and store it in a MongoDB database.
router.post('/saveSharesToMongo', function(req, res, next) {
  // Read request information
  console.log(`I received req.query = ${JSON.stringify(req.query)}`)
  
  const share_one = req.query.share_one
  const share_two = req.query.share_two
  const share_three = req.query.share_three
  console.log(`I received share_one = ${share_one}`)
  console.log(`I received share_two = ${share_two}`)
  console.log(`I received share_three = ${share_three}`)
  
  const to_store = {
    share_one: share_one,
    share_two: share_two,
    share_three: share_three,
  }
  console.log(`I computed to_store = ${JSON.stringify(to_store)}`)
  
  save_object_to_mongo(to_store)
  
  // // Generate random numbers
  // console.log('here3')
  // const random_float_uniform_zero_to_one = Math.random()
  // console.log(`random_float_uniform_zero_to_one = ${random_float_uniform_zero_to_one}`)
  
  // // Write memory to file
  // json_data = {k1: random_float_uniform_zero_to_one}
  // json_as_string = JSON.stringify(json_data)
  // write_to_file('random_float.txt', json_as_string)
  // console.log('here4')
  
  // Send response
  res.send({
    info: 'Thank you for participating in our research!',
  });
});

router.get('/', function(req, res, next) {
  console.log('Someone came to the home page!')
  
  const _sodium = require('libsodium-wrappers');
  (async() => {
    await _sodium.ready;
    const sodium = _sodium;
    
    const keypair = sodium.crypto_box_keypair();
    const public_key = keypair.publicKey;
    const private_key = keypair.privateKey;
    
    console.log(`${keypair.keyType}`);
    console.log(`${JSON.stringify(public_key)}`);
    console.log(`${JSON.stringify(private_key)}`);
    
    // Select message
    const m = 'message1'
    const mb = sodium.from_string(m)
    
    // Encrypt
    const c = sodium.crypto_box_seal(mb, public_key)
    
    // Decrypt
    const rb = sodium.crypto_box_seal_open(c, public_key, private_key)
    const r = sodium.to_string(rb)
    
    // Assert correctness
    if (m != r) {
      throw 'm != r'
    }
    
    console.log(m)
    console.log(r)
    console.log(sodium.to_base64(m))
    console.log(sodium.to_base64(r))
    console.log(sodium.from_base64(sodium.to_base64(m)))
    console.log(sodium.from_base64(sodium.to_base64(r)))
    console.log(sodium.to_string(sodium.from_base64(sodium.to_base64(m))))
    console.log(sodium.to_string(sodium.from_base64(sodium.to_base64(r))))
  })();
  
  res.render('index', { title: 'Express' });
});

module.exports = router;
