/**
 * webMUSHRA-style listening study for voice cloning evaluation.
 * Static, GitHub Pages compatible. Results POSTed to Google Sheets.
 */

class MushraStudy {
    constructor(config) {
        this.config = config;
        this.container = document.getElementById('page-container');
        this.results = [];
        this.currentPage = 0;
        this.pages = config.pages;
        this.participantId = this.generateId();
        this.startTime = Date.now();
        this.trialOrder = [];
    }

    generateId() {
        return 'P' + Date.now().toString(36) + Math.random().toString(36).substr(2, 5);
    }

    start() {
        this.render();
    }

    render() {
        const page = this.pages[this.currentPage];
        this.container.innerHTML = '';

        // Progress bar
        if (this.currentPage > 0) {
            const progress = document.createElement('div');
            progress.className = 'progress-bar';
            progress.innerHTML = `<div class="fill" style="width: ${(this.currentPage / this.pages.length) * 100}%"></div>`;
            this.container.appendChild(progress);
        }

        switch (page.type) {
            case 'consent': this.renderConsent(page); break;
            case 'screening': this.renderScreening(page); break;
            case 'headphone_check': this.renderHeadphoneCheck(page); break;
            case 'training': this.renderTraining(page); break;
            case 'mushra': this.renderMushraTrial(page); break;
            case 'completion': this.renderCompletion(page); break;
            default: this.renderGeneric(page);
        }
    }

    renderConsent(page) {
        const html = `
            <h1>${page.title || 'Informed Consent'}</h1>
            <div style="margin: 20px 0; max-height: 400px; overflow-y: auto; padding: 16px; background: #f8f9fa; border-radius: 4px;">
                ${page.content.map(p => `<p>${p}</p>`).join('')}
            </div>
            <label style="display: flex; align-items: center; gap: 8px; margin-top: 16px;">
                <input type="checkbox" id="consent-check">
                I have read and agree to participate in this study.
            </label>
            <button class="btn btn-next" id="consent-btn" disabled>Continue</button>
        `;
        this.container.innerHTML += html;
        const cb = document.getElementById('consent-check');
        const btn = document.getElementById('consent-btn');
        cb.addEventListener('change', () => { btn.disabled = !cb.checked; });
        btn.addEventListener('click', () => this.nextPage());
    }

    renderScreening(page) {
        let html = `<h1>Participant Information</h1>`;
        html += page.questions.map((q, i) => `
            <div class="screening-question">
                <label>${q.label}</label>
                ${q.type === 'select'
                    ? `<select id="screen-${i}" required>
                        <option value="">-- Select --</option>
                        ${q.options.map(o => `<option value="${o}">${o}</option>`).join('')}
                       </select>`
                    : `<input type="${q.type}" id="screen-${i}" placeholder="${q.placeholder || ''}" required>`
                }
            </div>
        `).join('');
        html += `<div id="screen-error" class="error-msg" style="display:none"></div>`;
        html += `<button class="btn btn-next" id="screen-btn">Continue</button>`;
        this.container.innerHTML += html;

        document.getElementById('screen-btn').addEventListener('click', () => {
            const answers = {};
            let valid = true;
            page.questions.forEach((q, i) => {
                const el = document.getElementById(`screen-${i}`);
                answers[q.id] = el.value;
                if (!el.value) valid = false;
            });
            if (!valid) {
                document.getElementById('screen-error').style.display = 'block';
                document.getElementById('screen-error').textContent = 'Please answer all questions.';
                return;
            }
            // Check exclusion criteria
            if (page.exclusions) {
                for (const exc of page.exclusions) {
                    if (answers[exc.id] === exc.value) {
                        this.container.innerHTML = `
                            <h1>Thank you</h1>
                            <p>${exc.message || 'Unfortunately you do not meet the criteria for this study.'}</p>
                        `;
                        return;
                    }
                }
            }
            this.results.push({ type: 'screening', answers, timestamp: Date.now() });
            this.nextPage();
        });
    }

    renderHeadphoneCheck(page) {
        let html = `
            <h1>Headphone Check</h1>
            <p>Please put on your headphones. You will hear a short tone. Which ear did you hear it in?</p>
            <div class="headphone-check">
                <audio id="hc-audio" src="${page.audio}" controls></audio>
                <div class="options" style="margin-top: 20px;">
                    <button class="hc-btn" data-answer="left">Left</button>
                    <button class="hc-btn" data-answer="right">Right</button>
                    <button class="hc-btn" data-answer="both">Both</button>
                </div>
                <div id="hc-feedback" style="margin-top: 16px;"></div>
            </div>
        `;
        this.container.innerHTML += html;

        document.querySelectorAll('.hc-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const answer = btn.dataset.answer;
                const correct = answer === page.correct_answer;
                this.results.push({ type: 'headphone_check', answer, correct, timestamp: Date.now() });
                if (correct) {
                    this.nextPage();
                } else {
                    document.getElementById('hc-feedback').innerHTML =
                        '<p class="warning">Incorrect. Please ensure you are wearing headphones and try again.</p>';
                }
            });
        });
    }

    renderTraining(page) {
        let html = `
            <h1>Practice Trial</h1>
            <p>This is a practice trial to familiarise you with the task. Your ratings here will not be recorded.</p>
            <div class="warning">Listen to the Reference first, then rate each sample on both scales.</div>
        `;
        html += this.buildTrialUI(page, true);
        this.container.innerHTML += html;
        this.attachSliderListeners();

        document.getElementById('next-btn').addEventListener('click', () => this.nextPage());
    }

    renderMushraTrial(page) {
        // Randomize condition order
        const condKeys = Object.keys(page.stimuli);
        const shuffled = this.shuffleArray([...condKeys]);

        let html = `<h2>Trial ${page.trialNumber} of ${page.totalTrials}</h2>`;
        html += `<p>${page.content || 'Rate each sample from 0 (bad) to 100 (excellent).'}</p>`;
        html += this.buildTrialUI(page, false, shuffled);
        this.container.innerHTML += html;
        this.attachSliderListeners();

        document.getElementById('next-btn').addEventListener('click', () => {
            const ratings = this.collectRatings(page, shuffled);
            if (!ratings) return;
            this.results.push({
                type: 'mushra',
                trialId: page.id,
                ratings,
                timestamp: Date.now()
            });
            this.nextPage();
        });
    }

    buildTrialUI(page, isTraining, condOrder) {
        const conditions = condOrder || Object.keys(page.stimuli);
        // Assign letter labels
        const labels = conditions.map((_, i) => String.fromCharCode(65 + i));

        let html = `
            <div class="trial-container">
                <div class="reference-player">
                    <h3>Reference</h3>
                    <audio controls src="${page.reference}"></audio>
                </div>
        `;

        // Naturalness scale
        html += `<div class="scale-header">Naturalness: How natural does the speech sound?</div>`;
        html += `<div class="scale-labels"><span>Bad (0)</span><span>Poor (20)</span><span>Fair (40)</span><span>Good (60)</span><span>Excellent (80-100)</span></div>`;
        conditions.forEach((cond, i) => {
            html += `
                <div class="condition-row">
                    <span class="label">Sample ${labels[i]}</span>
                    <audio controls src="${page.stimuli[cond]}"></audio>
                    <div class="slider-container">
                        <input type="range" min="0" max="100" value="50" id="nat-${cond}" data-scale="naturalness" data-cond="${cond}">
                        <span class="value" id="nat-val-${cond}">50</span>
                    </div>
                </div>
            `;
        });

        // Speaker similarity scale
        html += `<div class="scale-header">Speaker Match: How closely does the voice match the reference speaker?</div>`;
        html += `<div class="scale-labels"><span>Bad (0)</span><span>Poor (20)</span><span>Fair (40)</span><span>Good (60)</span><span>Excellent (80-100)</span></div>`;
        conditions.forEach((cond, i) => {
            html += `
                <div class="condition-row">
                    <span class="label">Sample ${labels[i]}</span>
                    <audio controls src="${page.stimuli[cond]}"></audio>
                    <div class="slider-container">
                        <input type="range" min="0" max="100" value="50" id="sim-${cond}" data-scale="similarity" data-cond="${cond}">
                        <span class="value" id="sim-val-${cond}">50</span>
                    </div>
                </div>
            `;
        });

        html += `</div>`;
        html += `<div id="trial-error" class="error-msg" style="display:none"></div>`;
        html += `<button class="btn btn-next" id="next-btn">Next</button>`;
        return html;
    }

    attachSliderListeners() {
        document.querySelectorAll('input[type="range"]').forEach(slider => {
            const valEl = document.getElementById(slider.id.replace('nat-', 'nat-val-').replace('sim-', 'sim-val-'));
            if (valEl) {
                slider.addEventListener('input', () => { valEl.textContent = slider.value; });
            }
        });
    }

    collectRatings(page, condOrder) {
        const ratings = {};
        for (const cond of condOrder) {
            const nat = document.getElementById(`nat-${cond}`);
            const sim = document.getElementById(`sim-${cond}`);
            if (!nat || !sim) return null;
            ratings[cond] = {
                naturalness: parseInt(nat.value),
                similarity: parseInt(sim.value)
            };
        }
        return ratings;
    }

    renderCompletion(page) {
        // Submit results
        this.submitResults();

        const code = page.completionCode || 'VCMUSHRA2026';
        this.container.innerHTML = `
            <div class="completion">
                <h1>Study Complete</h1>
                <p>Thank you for participating. Your responses have been recorded.</p>
                <p>Your Prolific completion code:</p>
                <div class="code">${code}</div>
                <p style="margin-top: 20px; color: #666;">You may now close this window.</p>
            </div>
        `;
    }

    renderGeneric(page) {
        let html = `<h1>${page.title || page.name || ''}</h1>`;
        if (page.content) {
            const content = Array.isArray(page.content) ? page.content : [page.content];
            html += content.map(p => p ? `<p>${p}</p>` : '<br>').join('');
        }
        html += `<button class="btn btn-next" id="next-btn">Continue</button>`;
        this.container.innerHTML += html;
        document.getElementById('next-btn').addEventListener('click', () => this.nextPage());
    }

    nextPage() {
        this.currentPage++;
        if (this.currentPage >= this.pages.length) {
            this.currentPage = this.pages.length - 1;
        }
        this.render();
        window.scrollTo(0, 0);
    }

    shuffleArray(arr) {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
        return arr;
    }

    async submitResults() {
        const payload = {
            participantId: this.participantId,
            startTime: this.startTime,
            endTime: Date.now(),
            durationMs: Date.now() - this.startTime,
            userAgent: navigator.userAgent,
            results: this.results
        };

        // Try Google Sheets endpoint
        const endpoint = this.config.resultsEndpoint;
        if (endpoint) {
            try {
                await fetch(endpoint, {
                    method: 'POST',
                    mode: 'no-cors',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
            } catch (e) {
                console.error('Failed to submit results:', e);
            }
        }

        // Always save locally as backup
        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `results_${this.participantId}.json`;
        // Don't auto-download, just log
        console.log('Results payload:', payload);
        localStorage.setItem(`mushra_results_${this.participantId}`, JSON.stringify(payload));
    }
}
