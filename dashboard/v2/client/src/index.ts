import Heatmap from './heatmap';


class Tabs {
  _heatmaps: Heatmap[];

  constructor() {
    this._heatmaps = [];

    // Fetch a list of test prefixes from the backend, then initialize
    fetch('/api/get_test_prefixes', {
      method: 'POST',
    }).then(resp => resp.json()).then(resp => this._init(resp));
  }

  _init(testPrefixes: string[]) {
    // Initialize DOM and create Heatmap objects
    let tabButtonsWrapper = document.getElementById('tab-buttons')!;
    let tabContentsWrapper = document.getElementById('tab-contents')!;
    for (let i = 0; i < testPrefixes.length; i++) {
      let tabContentId = `tab${i}`;
      let spinnerId = `spinner${i}`;
      let heatmapId = `heatmap${i}`;

      tabButtonsWrapper.innerHTML += `
        <button class="tab-button" id="${i}">${testPrefixes[i]}</button>
      `;

      tabContentsWrapper.innerHTML += `
        <div class="tab-content" id="${tabContentId}">
          <div class="spinner-wrapper" id="${spinnerId}">
            <div class="spinner">
              <div></div><div></div><div></div><div></div>
            </div>
          </div>
          <div id="${heatmapId}"></div>
        </div>
      `;

      this._heatmaps.push(new Heatmap(heatmapId, spinnerId, testPrefixes[i]));
    }

    // Add event listeners to the tab buttons
    let tabBtns = document.getElementsByClassName('tab-button');
    for (let btn of tabBtns) {
      (btn as HTMLElement).onclick = (e) => {
        e.preventDefault();
        let target = e.currentTarget! as HTMLElement;
        this._onTabClick(target.id);
        return false;
      }
    }

    // Select the first tab
    this._onTabClick('0');
  }

  _onTabClick(id: string) {
    // Hide all tabs
    let tabs = document.getElementsByClassName('tab-content');
    for (let tab of tabs) {
      (tab as HTMLElement).style.display = 'none';
    }

    // Unclick all tab buttons
    let tabButtons = document.getElementsByClassName('tab-button');
    for (let btn of tabButtons) {
      btn.classList.remove('active');
    }

    // Click target button and reveal target tab
    document.getElementById(id)!.classList.add('active');
    document.getElementById(`tab${id}`)!.style.display = "block";

    // Load heatmap data and render (if not already loaded)
    let heatmap = this._heatmaps[parseInt(id)];
    if (!heatmap.isLoaded()) {
      heatmap.loadDataAndRender();
    }
  }
}


let _ = new Tabs();


