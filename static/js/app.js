document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const algorithmSelect = document.getElementById('algorithm');
    const graphSelect = document.getElementById('graph-file');
    const parallelParams = document.getElementById('parallel-params');
    const threadsGroup = document.getElementById('threads-group');
    const procsGroup = document.getElementById('procs-group');
    const runForm = document.getElementById('run-form');
    
    const consoleOutput = document.getElementById('console-output');
    const loadingOverlay = document.getElementById('loading-overlay');
    const submitBtn = document.getElementById('submit-btn');
    
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    
    const statsContainer = document.getElementById('stats-container');
    const statTime = document.getElementById('stat-time');
    const statAlg = document.getElementById('stat-alg');

    // Graph Elements
    const graphNetwork = document.getElementById('graph-network');
    const graphWarning = document.getElementById('graph-warning');
    const graphStats = document.getElementById('graph-stats');
    let network = null;

    // 1. Fetch available graphs on load
    fetchGraphs();

    async function fetchGraphs() {
        try {
            const response = await fetch('/api/graphs');
            const data = await response.json();

            if (data.success) {
                graphSelect.innerHTML = '';
                if (data.graphs.length === 0) {
                    graphSelect.innerHTML = '<option value="">No graphs found in graphs/ directory</option>';
                    return;
                }
                
                // Sort array so small -> medium -> large -> custom
                const predefined = ['tiny.txt', 'small.txt', 'medium.txt', 'large.txt'];
                data.graphs.sort((a, b) => {
                    const idxA = predefined.indexOf(a);
                    const idxB = predefined.indexOf(b);
                    if (idxA !== -1 && idxB !== -1) return idxA - idxB;
                    if (idxA !== -1) return -1;
                    if (idxB !== -1) return 1;
                    return a.localeCompare(b);
                });

                data.graphs.forEach(graph => {
                    const option = document.createElement('option');
                    option.value = graph;
                    option.textContent = graph;
                    graphSelect.appendChild(option);
                });
                
                // Load first graph right away
                if (data.graphs.length > 0) {
                    fetchAndDrawGraph(data.graphs[0]);
                }

            } else {
                console.error("Failed to load graphs:", data.error);
                graphSelect.innerHTML = '<option value="">Error loading graphs</option>';
            }
        } catch (error) {
            console.error("Network error:", error);
            graphSelect.innerHTML = '<option value="">Error connecting to server</option>';
        }
    }

    // 2. Handle dynamic form inputs based on algorithm selection
    function updateFormVisibility() {
        const alg = algorithmSelect.value;
        const needsThreads = ['openmp', 'hybrid'].includes(alg);
        const needsProcs = ['mpi', 'hybrid'].includes(alg);

        if (needsThreads || needsProcs) {
            parallelParams.classList.remove('hidden');
            
            if (needsThreads) {
                threadsGroup.classList.remove('hidden');
                document.getElementById('threads').required = true;
            } else {
                threadsGroup.classList.add('hidden');
                document.getElementById('threads').required = false;
            }

            if (needsProcs) {
                procsGroup.classList.remove('hidden');
                document.getElementById('processes').required = true;
            } else {
                procsGroup.classList.add('hidden');
                document.getElementById('processes').required = false;
            }
        } else {
            parallelParams.classList.add('hidden');
            document.getElementById('threads').required = false;
            document.getElementById('processes').required = false;
        }
    }

    algorithmSelect.addEventListener('change', updateFormVisibility);
    updateFormVisibility(); // Init

    // 3. Listen for Graph File Selection Changes
    graphSelect.addEventListener('change', () => {
        if(graphSelect.value) {
            fetchAndDrawGraph(graphSelect.value);
        }
    });

    // 4. Fetch Graph Data and Draw
    async function fetchAndDrawGraph(filename) {
        try {
            graphStats.textContent = "Loading graph structure...";
            const response = await fetch(`/api/graph_data?file=${encodeURIComponent(filename)}`);
            const data = await response.json();

            if (data.success) {
                graphStats.textContent = `${data.totalV} Nodes | ${data.totalE} Edges`;
                
                if (data.truncated) {
                    graphWarning.classList.remove('hidden');
                } else {
                    graphWarning.classList.add('hidden');
                }

                drawGraph(data.nodes, data.edges);
            } else {
                graphStats.textContent = "Error loading graph";
            }
        } catch(error) {
            console.error("Error drawing graph:", error);
            graphStats.textContent = "Error";
        }
    }

    function drawGraph(nodesData, edgesData) {
        // Destroy old network if it exists
        if (network !== null) {
            network.destroy();
            network = null;
        }

        const nodes = new vis.DataSet(nodesData);
        const edges = new vis.DataSet(edgesData);
        
        const data = { nodes, edges };
        
        const options = {
            nodes: {
                shape: 'dot',
                size: 16,
                font: { color: '#ffffff', face: 'Inter', size: 12 },
                color: {
                    background: '#2f81f7',
                    border: '#1f6feb',
                    highlight: { background: '#ab68ff', border: '#ab68ff' }
                },
                borderWidth: 2,
                shadow: true
            },
            edges: {
                width: 1.5,
                color: { color: '#30363d', highlight: '#8b949e' },
                font: { color: '#8b949e', face: 'Inter', size: 11, align: 'top' },
                arrows: { to: { enabled: true, scaleFactor: 0.8 } },
                smooth: { type: 'continuous' }
            },
            physics: {
                barnesHut: { 
                    gravitationalConstant: -2000, 
                    springConstant: 0.04,
                    springLength: 95
                },
                stabilization: { iterations: 150 }
            },
            interaction: {
                hover: true,
                tooltipDelay: 200,
                zoomView: true
            }
        };

        network = new vis.Network(graphNetwork, data, options);
    }

    // 5. Handle form submission
    runForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Prepare Payload
        const payload = {
            algorithm: algorithmSelect.value,
            graph_file: graphSelect.value,
            source: parseInt(document.getElementById('source').value),
            threads: parseInt(document.getElementById('threads').value),
            processes: parseInt(document.getElementById('processes').value)
        };

        // Update UI for loading state
        submitBtn.disabled = true;
        submitBtn.querySelector('span').textContent = 'Running...';
        loadingOverlay.classList.add('active');
        consoleOutput.textContent = '';
        
        statusDot.className = 'dot running';
        statusText.textContent = `Running ${payload.algorithm.toUpperCase()}...`;
        statsContainer.classList.add('hidden');

        try {
            const response = await fetch('/api/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const data = await response.json();

            // Reset UI
            loadingOverlay.classList.remove('active');
            submitBtn.disabled = false;
            submitBtn.querySelector('span').textContent = 'Execute Run';
            
            if (data.success) {
                statusDot.className = 'dot success';
                statusText.textContent = 'Execution Complete';
                
                // Display output
                let filteredOutput = data.stdout;
                // If it's too long, truncate it
                if (filteredOutput.length > 50000) {
                     filteredOutput = filteredOutput.substring(0, 50000) + "\n\n...[OUTPUT TRUNCATED - TOO LONG TO DISPLAY]...";
                }
                consoleOutput.textContent = filteredOutput;

                // Update Stats
                statsContainer.classList.remove('hidden');
                
                let timeVal = data.time;
                // If server didn't explicitly return time, try matching stdout
                if (!timeVal) {
                    const match = data.stdout.match(/Execution time\s*:\s*([\d\.]+)/);
                    if (match) timeVal = match[1];
                }

                statTime.textContent = timeVal ? `${timeVal} s` : "N/A";
                
                let algName = algorithmSelect.options[algorithmSelect.selectedIndex].text;
                if (payload.algorithm === 'openmp') algName += ` (${payload.threads} threads)`;
                if (payload.algorithm === 'mpi') algName += ` (${payload.processes} procs)`;
                if (payload.algorithm === 'hybrid') algName += ` (${payload.processes}P × ${payload.threads}T)`;
                
                statAlg.textContent = algName;
            } else {
                throw new Error(data.error);
            }

        } catch (error) {
            loadingOverlay.classList.remove('active');
            submitBtn.disabled = false;
            submitBtn.querySelector('span').textContent = 'Execute Run';
            
            statusDot.className = 'dot error';
            statusText.textContent = 'Execution Failed';
            
            // Format error properly
            consoleOutput.innerHTML = `<span style="color: var(--accent-red)">ERROR: ${error.message}</span>`;
            if (error.stderr) {
                 consoleOutput.innerHTML += `\n\n<span style="color: #c9d1d9">STDERR:</span>\n${error.stderr}`;
            }
        }
    });

    // 6. Navigation and Benchmarks Loading
    const navExecute = document.getElementById('nav-execute');
    const navBenchmarks = document.getElementById('nav-benchmarks');
    const viewExecute = document.getElementById('view-execute');
    const viewBenchmarks = document.getElementById('view-benchmarks');

    navExecute.addEventListener('click', (e) => {
        e.preventDefault();
        navExecute.classList.add('active');
        navBenchmarks.classList.remove('active');
        viewExecute.classList.remove('hidden');
        viewBenchmarks.classList.add('hidden');
    });

    navBenchmarks.addEventListener('click', (e) => {
        e.preventDefault();
        navBenchmarks.classList.add('active');
        navExecute.classList.remove('active');
        viewBenchmarks.classList.remove('hidden');
        viewExecute.classList.add('hidden');
        loadBenchmarks();
    });

    let benchmarkData = [];
    const benchmarkFilter = document.getElementById('benchmark-filter');

    benchmarkFilter.addEventListener('change', () => {
        renderBenchmarkTable(benchmarkFilter.value);
    });

    async function loadBenchmarks() {
        const statusEl = document.getElementById('benchmark-status');
        const tbody = document.getElementById('benchmark-table-body');
        
        // Don't reload if already loaded
        if (benchmarkData.length > 0) {
            renderBenchmarkTable(benchmarkFilter.value);
            return;
        }

        statusEl.textContent = 'Loading data...';
        try {
            const response = await fetch('/api/benchmarks');
            const data = await response.json();
            
            if (data.success) {
                benchmarkData = data.data;
                statusEl.textContent = `${benchmarkData.length} results loaded`;

                // Populate filter dropdown
                const uniqueGraphs = [...new Set(benchmarkData.map(r => r.graph || '-'))];
                const sizeOrder = ['tiny', 'small', 'medium', 'large', 'xlarge_pos', 'xxlarge_pos'];
                uniqueGraphs.sort((a, b) => {
                    const idxA = sizeOrder.indexOf(a);
                    const idxB = sizeOrder.indexOf(b);
                    if (idxA !== -1 && idxB !== -1) return idxA - idxB;
                    if (idxA !== -1) return -1;
                    if (idxB !== -1) return 1;
                    return a.localeCompare(b);
                });

                // Keep the "All Sizes" option and add the rest
                benchmarkFilter.innerHTML = '<option value="all">All Sizes</option>';
                uniqueGraphs.forEach(g => {
                    const opt = document.createElement('option');
                    opt.value = g;
                    opt.textContent = g.toUpperCase();
                    benchmarkFilter.appendChild(opt);
                });

                renderBenchmarkTable('all');
            } else {
                statusEl.textContent = `Error: ${data.error}`;
                tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--accent-red)">Failed to load data: ${data.error}</td></tr>`;
            }
        } catch (err) {
            console.error('Error loading benchmarks:', err);
            statusEl.textContent = 'Network error';
            tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--accent-red)">Network error loading data</td></tr>`;
        }
    }

    function renderBenchmarkTable(filterGraph) {
        const tbody = document.getElementById('benchmark-table-body');
        tbody.innerHTML = '';
        
        let filteredData = benchmarkData;
        if (filterGraph !== 'all') {
            filteredData = benchmarkData.filter(r => (r.graph || '-') === filterGraph);
        }

        if (filteredData.length === 0) {
            tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted)">No results found.</td></tr>`;
            return;
        }

        // Group by graph size
        const groupedData = {};
        filteredData.forEach(row => {
            const g = row.graph || '-';
            if (!groupedData[g]) groupedData[g] = [];
            groupedData[g].push(row);
        });

        const sizeOrder = ['tiny', 'small', 'medium', 'large', 'xlarge_pos', 'xxlarge_pos'];
        const sortedGraphs = Object.keys(groupedData).sort((a, b) => {
            const idxA = sizeOrder.indexOf(a);
            const idxB = sizeOrder.indexOf(b);
            if (idxA !== -1 && idxB !== -1) return idxA - idxB;
            if (idxA !== -1) return -1;
            if (idxB !== -1) return 1;
            return a.localeCompare(b);
        });

        sortedGraphs.forEach(graphSize => {
            const headerTr = document.createElement('tr');
            headerTr.className = 'benchmark-group-header';
            headerTr.innerHTML = `<td colspan="4" style="background-color: rgba(47, 129, 247, 0.1); color: var(--accent-blue); padding: 10px 16px;">
                <i class="fa-solid fa-folder-tree" style="margin-right: 8px;"></i> Graph Size: <strong style="text-transform: uppercase;">${graphSize}</strong>
            </td>`;
            tbody.appendChild(headerTr);

            groupedData[graphSize].forEach(row => {
                const tr = document.createElement('tr');
                
                let speedupClass = 'speedup-low';
                const speedupVal = parseFloat(row.speedup);
                if (!isNaN(speedupVal)) {
                    if (speedupVal >= 2.0) speedupClass = 'speedup-high';
                    else if (speedupVal > 1.0) speedupClass = 'speedup-med';
                }

                tr.innerHTML = `
                    <td><span style="color: var(--accent-purple)">${(row.version || '').toUpperCase()}</span></td>
                    <td>${row.config || '-'}</td>
                    <td style="font-family: monospace">${row.time_sec || '-'}</td>
                    <td class="${speedupClass}">${row.speedup && row.speedup !== 'N/A' ? row.speedup + 'x' : 'N/A'}</td>
                `;
                tbody.appendChild(tr);
            });
        });
    }
});
