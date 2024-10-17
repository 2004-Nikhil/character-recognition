var radarChart;
var canvas, ctx;
var mouseX, mouseY, mouseDown = 0;
var touchX, touchY;
var confidencePercentages = Array(47).fill(0);
var debounceTimeout;

function init() {
    canvas = document.getElementById('sketchpad');
    ctx = canvas.getContext('2d');
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (ctx) {
        canvas.addEventListener('mousedown', sketchpad_mouseDown, false);
        canvas.addEventListener('mousemove', sketchpad_mouseMove, false);
        window.addEventListener('mouseup', sketchpad_mouseUp, false);
        canvas.addEventListener('touchstart', sketchpad_touchStart, false);
        canvas.addEventListener('touchmove', sketchpad_touchMove, false);
    }
}

// Draw function to draw on canvas
function draw(ctx, x, y, size, isDown) {
    if (isDown) {
        ctx.beginPath();
        ctx.strokeStyle = "white";
        ctx.lineWidth = '15';
        ctx.lineJoin = ctx.lineCap = 'round';
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.closePath();
        ctx.stroke();
        clearTimeout(debounceTimeout);
        debounceTimeout = setTimeout(predict, 100); 
    }
    lastX = x;
    lastY = y;
}

// Event handlers
function sketchpad_mouseDown(e) { // Pass event object
    mouseDown = 1;
    draw(ctx, mouseX, mouseY, 12, false);
}

function sketchpad_mouseUp(e) { // Pass event object
    mouseDown = 0;
}

function sketchpad_mouseMove(e) { // Pass event object
    getMousePos(e);  // Pass the event to getMousePos
    if (mouseDown == 1) {
        draw(ctx, mouseX, mouseY, 12, true);
    }
}

function getMousePos(e) {
    if (e.offsetX) {
        mouseX = e.offsetX;
        mouseY = e.offsetY;
    } else if (e.layerX) {
        mouseX = e.layerX;
        mouseY = e.layerY;
    }
}

function sketchpad_touchStart(e) { // Pass event object
    getTouchPos(e);  // Pass event
    draw(ctx, touchX, touchY, 12, false);
    e.preventDefault(); // Use the passed event object
}

function sketchpad_touchMove(e) { // Pass event object
    getTouchPos(e);  // Pass event
    draw(ctx, touchX, touchY, 12, true);
    e.preventDefault(); // Use the passed event object
}

function getTouchPos(e) {
    if (e.touches) {
        if (e.touches.length == 1) {
            var touch = e.touches[0];
            touchX = touch.pageX - touch.target.offsetLeft;
            touchY = touch.pageY - touch.target.offsetTop;
        }
    }
}

function predict() {
    var imageData = canvas.toDataURL();
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image_data: imageData })
    })
    .then(response => response.json())
    .then(data => {
        updateRadarChart(data.results);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

// Clear canvas function
document.getElementById('clear_button').addEventListener("click",
    function () {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        updateRadarChart(Array(47).fill(0));
    });

// Output
function updateRadarChart(confidenceArray) {
    var labels = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','d','e','f','g','h','n','q','r','t'];

    if (!radarChart) {
        var chartData = {
            labels: labels,
            datasets: [{
                label: 'Confidence Percentage',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                pointBackgroundColor: 'rgba(54, 162, 235, 1)',
                data: confidenceArray.map(confidence => confidence * 100)
            }]
        };

        var chartOptions = {
            scale: {
                ticks: {
                    beginAtZero: true,
                    min: 0,
                    max: 100,
                    callback: function(value) {
                        return value + '%';
                    }
                }
            }
        };

        radarChart = new Chart(document.getElementById('radarChart'), {
            type: 'radar',
            data: chartData,
            options: chartOptions
        });
    } else {
        radarChart.data.datasets[0].data = confidenceArray.map(confidence => confidence * 100);
        radarChart.update();
    }
}

document.addEventListener("DOMContentLoaded", function() {
    init();
    updateRadarChart(confidencePercentages);
});
