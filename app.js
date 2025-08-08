document.getElementById('query-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = document.getElementById('query-input').value;
    const responseArea = document.getElementById('response-area');
    
    responseArea.innerHTML = '<p>Processing your request...</p>';
    
    try {
        const response = await fetch('http://localhost:5000/api/research', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        responseArea.innerHTML = `
            <h3>${data.topic || 'Response'}</h3>
            <p>${data.summary}</p>
            ${data.sources ? `<h4>Sources:</h4><ul>${data.sources.map(s => `<li>${s}</li>`).join('')}</ul>` : ''}
        `;
    } catch (error) {
        responseArea.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    }
});