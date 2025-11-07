let dataset = null;
let selectedFeatures = [];
let targetColumn = null;

// File upload handler
document.getElementById('csvFile').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        document.getElementById('fileName').textContent = file.name;
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            complete: function(results) {
                dataset = results.data.filter(row => Object.values(row).some(val => val !== null && val !== ''));
                cleanData();
                showFeatureSelection();
            }
        });
    }
});

// Clean missing values
function cleanData() {
    const columns = Object.keys(dataset[0]);
    
    columns.forEach(col => {
        const values = dataset.map(row => row[col]);
        const validValues = values.filter(v => v !== null && v !== '' && v !== undefined);
        const isNumeric = validValues.every(v => typeof v === 'number');
        
        // Fill missing values with mean (numeric) or mode (categorical)
        const fillValue = isNumeric 
            ? validValues.reduce((a, b) => a + b, 0) / validValues.length
            : validValues.sort((a,b) => 
                validValues.filter(v => v === a).length - validValues.filter(v => v === b).length
              ).pop();
        
        dataset.forEach(row => {
            if (row[col] === null || row[col] === '' || row[col] === undefined) {
                row[col] = fillValue;
            }
        });
    });
}

// Show feature selection UI
function showFeatureSelection() {
    const columns = Object.keys(dataset[0]);
    const selectionDiv = document.getElementById('featureSelection');
    
    document.getElementById('dataStats').innerHTML = 
        `<strong>Rows:</strong> ${dataset.length} | <strong>Columns:</strong> ${columns.length}`;
    document.getElementById('dataInfo').style.display = 'block';
    
    // Create data preview table (first 5 rows)
    const previewRows = dataset.slice(0, 5);
    const tableHTML = `
        <div class="table-container">
            <table class="data-preview">
                <thead>
                    <tr>${columns.map(col => `<th>${col}</th>`).join('')}</tr>
                </thead>
                <tbody>
                    ${previewRows.map(row => `
                        <tr>${columns.map(col => `<td>${row[col]}</td>`).join('')}</tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
    
    // Create checkboxes for features
    const featuresHTML = columns.map(col => `
        <label class="feature-checkbox">
            <input type="checkbox" value="${col}" class="feature-check">
            ${col}
        </label>
    `).join('');
    
    // Create dropdown for target
    const targetHTML = `
        <select id="targetSelect">
            <option value="">-- Select Target --</option>
            ${columns.map(col => `<option value="${col}">${col}</option>`).join('')}
        </select>
    `;
    
    selectionDiv.innerHTML = `
        <h3>Data Preview (First 5 Rows)</h3>
        ${tableHTML}
        <h3>Select Features (Independent Variables)</h3>
        <div class="features-grid">${featuresHTML}</div>
        <h3>Select Target (Dependent Variable)</h3>
        ${targetHTML}
    `;
    
    selectionDiv.style.display = 'block';
    document.getElementById('confirmBtn').style.display = 'block';
}

// Confirm selection and train
document.getElementById('confirmBtn').addEventListener('click', function() {
    const checkboxes = document.querySelectorAll('.feature-check:checked');
    selectedFeatures = Array.from(checkboxes).map(cb => cb.value);
    targetColumn = document.getElementById('targetSelect').value;
    
    if (selectedFeatures.length === 0 || !targetColumn) {
        alert('Please select at least one feature and a target variable');
        return;
    }
    
    if (selectedFeatures.includes(targetColumn)) {
        alert('Target variable cannot be a feature');
        return;
    }
    
    document.getElementById('featureSelection').style.display = 'none';
    document.getElementById('confirmBtn').style.display = 'none';
    trainModels();
});

// Train all models
async function trainModels() {
    document.getElementById('loading').style.display = 'block';
    await new Promise(resolve => setTimeout(resolve, 100));
    
    try {
        const { X, y } = prepareData();
        const splitIdx = Math.floor(X.length * 0.8);
        
        const XTrain = X.slice(0, splitIdx);
        const yTrain = y.slice(0, splitIdx);
        const XTest = X.slice(splitIdx);
        const yTest = y.slice(splitIdx);
        
        const results = {
            logistic: trainLogistic(XTrain, yTrain, XTest, yTest),
            knn: trainKNN(XTrain, yTrain, XTest, yTest),
            tree: trainTree(XTrain, yTrain, XTest, yTest),
            randomForest: trainRandomForest(XTrain, yTrain, XTest, yTest),
            naiveBayes: trainNaiveBayes(XTrain, yTrain, XTest, yTest)
        };
        
        displayResults(results);
    } catch (error) {
        alert('Error: ' + error.message);
    }
    
    document.getElementById('loading').style.display = 'none';
}

// Prepare data
function prepareData() {
    // Encode categorical variables
    const encoders = {};
    selectedFeatures.forEach(col => {
        const values = dataset.map(row => row[col]);
        if (typeof values[0] === 'string') {
            const unique = [...new Set(values)];
            encoders[col] = Object.fromEntries(unique.map((v, i) => [v, i]));
        }
    });
    
    // Encode target
    const targetValues = dataset.map(row => row[targetColumn]);
    const uniqueTargets = [...new Set(targetValues)];
    const targetEncoder = Object.fromEntries(uniqueTargets.map((v, i) => [v, i]));
    
    // Build X and y
    const X = dataset.map(row => 
        selectedFeatures.map(col => {
            const val = row[col];
            return encoders[col] ? encoders[col][val] : val;
        })
    );
    
    const y = dataset.map(row => targetEncoder[row[targetColumn]]);
    
    // Normalize X
    return { X: normalize(X), y };
}

// Normalize features
function normalize(X) {
    const numFeatures = X[0].length;
    const mins = Array(numFeatures).fill(Infinity);
    const maxs = Array(numFeatures).fill(-Infinity);
    
    X.forEach(row => row.forEach((val, i) => {
        if (val < mins[i]) mins[i] = val;
        if (val > maxs[i]) maxs[i] = val;
    }));
    
    return X.map(row => row.map((val, i) => {
        const range = maxs[i] - mins[i];
        return range === 0 ? 0 : (val - mins[i]) / range;
    }));
}

// Logistic Regression
function trainLogistic(XTrain, yTrain, XTest, yTest) {
    const n = XTrain[0].length;
    let w = Array(n).fill(0);
    let b = 0;
    
    for (let epoch = 0; epoch < 100; epoch++) {
        for (let i = 0; i < XTrain.length; i++) {
            let z = b;
            for (let j = 0; j < n; j++) z += w[j] * XTrain[i][j];
            const pred = 1 / (1 + Math.exp(-z));
            const err = pred - yTrain[i];
            
            b -= 0.1 * err;
            for (let j = 0; j < n; j++) w[j] -= 0.1 * err * XTrain[i][j];
        }
    }
    
    let correct = 0;
    for (let i = 0; i < XTest.length; i++) {
        let z = b;
        for (let j = 0; j < n; j++) z += w[j] * XTest[i][j];
        const pred = (1 / (1 + Math.exp(-z))) > 0.5 ? 1 : 0;
        if (pred === yTest[i]) correct++;
    }
    
    return correct / XTest.length;
}

// KNN
function trainKNN(XTrain, yTrain, XTest, yTest) {
    const knn = new ML.KNN(XTrain, yTrain, { k: 3 });
    let correct = 0;
    
    for (let i = 0; i < XTest.length; i++) {
        if (knn.predict(XTest[i]) === yTest[i]) correct++;
    }
    
    return correct / XTest.length;
}

// Decision Tree
function trainTree(XTrain, yTrain, XTest, yTest) {
    const tree = new ML.DecisionTreeClassifier({ maxDepth: 10 });
    tree.train(XTrain, yTrain);
    
    let correct = 0;
    for (let i = 0; i < XTest.length; i++) {
        if (tree.predict([XTest[i]])[0] === yTest[i]) correct++;
    }
    
    return correct / XTest.length;
}

// Random Forest
function trainRandomForest(XTrain, yTrain, XTest, yTest) {
    const rf = new ML.RandomForestClassifier({ nEstimators: 10, maxDepth: 10 });
    rf.train(XTrain, yTrain);
    
    let correct = 0;
    for (let i = 0; i < XTest.length; i++) {
        if (rf.predict([XTest[i]])[0] === yTest[i]) correct++;
    }
    
    return correct / XTest.length;
}

// Naive Bayes (Simple implementation)
function trainNaiveBayes(XTrain, yTrain, XTest, yTest) {
    // Get unique classes
    const classes = [...new Set(yTrain)];
    
    // Calculate class priors and statistics
    const classStats = {};
    classes.forEach(c => {
        const classData = XTrain.filter((_, i) => yTrain[i] === c);
        const numFeatures = XTrain[0].length;
        
        classStats[c] = {
            prior: classData.length / XTrain.length,
            means: [],
            stds: []
        };
        
        // Calculate mean and std for each feature
        for (let f = 0; f < numFeatures; f++) {
            const values = classData.map(row => row[f]);
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
            const std = Math.sqrt(variance) + 1e-9; // Add small value to avoid division by zero
            
            classStats[c].means.push(mean);
            classStats[c].stds.push(std);
        }
    });
    
    // Predict function using Gaussian probability
    function predict(x) {
        let bestClass = classes[0];
        let maxProb = -Infinity;
        
        classes.forEach(c => {
            let logProb = Math.log(classStats[c].prior);
            
            for (let f = 0; f < x.length; f++) {
                const mean = classStats[c].means[f];
                const std = classStats[c].stds[f];
                const exponent = -Math.pow(x[f] - mean, 2) / (2 * Math.pow(std, 2));
                const prob = Math.exp(exponent) / (std * Math.sqrt(2 * Math.PI));
                logProb += Math.log(prob + 1e-9);
            }
            
            if (logProb > maxProb) {
                maxProb = logProb;
                bestClass = c;
            }
        });
        
        return bestClass;
    }
    
    // Evaluate
    let correct = 0;
    for (let i = 0; i < XTest.length; i++) {
        if (predict(XTest[i]) === yTest[i]) correct++;
    }
    
    return correct / XTest.length;
}

// Display results
function displayResults(results) {
    document.getElementById('logisticAccuracy').textContent = (results.logistic * 100).toFixed(1) + '%';
    document.getElementById('knnAccuracy').textContent = (results.knn * 100).toFixed(1) + '%';
    document.getElementById('treeAccuracy').textContent = (results.tree * 100).toFixed(1) + '%';
    document.getElementById('rfAccuracy').textContent = (results.randomForest * 100).toFixed(1) + '%';
    document.getElementById('nbAccuracy').textContent = (results.naiveBayes * 100).toFixed(1) + '%';
    document.getElementById('results').style.display = 'block';
    
    drawChart(results);
}

// Draw chart
function drawChart(results) {
    const canvas = document.getElementById('chartCanvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 800;
    canvas.height = 350;
    
    const accuracies = [results.logistic, results.knn, results.tree, results.randomForest, results.naiveBayes];
    const labels = ['Logistic', 'KNN', 'Tree', 'Random\nForest', 'Naive\nBayes'];
    const colors = ['#f39c12', '#e74c3c', '#e67e22', '#f1c40f', '#d35400'];
    
    ctx.fillStyle = '#2c3e50';
    ctx.fillRect(0, 0, 800, 350);
    
    // Draw bars
    const barWidth = 70;
    const spacing = 80;
    const startX = 80;
    
    accuracies.forEach((acc, i) => {
        const x = startX + i * (barWidth + spacing);
        const h = acc * 250;
        const y = 300 - h;
        
        ctx.fillStyle = colors[i];
        ctx.fillRect(x, y, barWidth, h);
        
        ctx.fillStyle = '#ecf0f1';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        
        // Handle multi-line labels
        const labelLines = labels[i].split('\n');
        labelLines.forEach((line, idx) => {
            ctx.fillText(line, x + barWidth / 2, 320 + idx * 14);
        });
        
        ctx.font = 'bold 14px Arial';
        ctx.fillText((acc * 100).toFixed(1) + '%', x + barWidth / 2, y - 10);
    });
}
