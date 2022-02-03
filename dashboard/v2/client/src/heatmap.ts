import * as d3 from 'd3';


const MARGIN = {top: 80, right: 25, bottom: 30, left: 400};
const WIDTH = 1400 - MARGIN.left - MARGIN.right;
const HEIGHT_PER_TEST = 30;


type TestRun = {
  date: string,
  test: string,
  status: string,
};

type HeatmapData = [TestRun[], string[], string[]];


// Returns the color of a test run given its status
function getSquareColor(status: string): string {
  return status === 'success' ? '#02cf17' : '#a10606';
}


// Converts pandas-style data from the backend into the format expected by D3
function processData(data: any): HeatmapData {
  let runs = [];
  let dates = new Set<string>();
  let tests = new Set<string>();
  for (const key in data['job_status']) {
    let date = data['run_date'][key];
    dates.add(date);

    let test = data['test_name'][key];
    tests.add(test);

    runs.push({
      'date': date,
      'test': test,
      'status': data['job_status'][key],
    });
  }

  // Convert the sets to arrays and reverse sort
  let dateList = [...dates].sort().reverse();
  let testList = [...tests].sort().reverse();

  return [runs, dateList, testList];
}


export default class Heatmap {
  heatmapId: string;
  spinnerId: string;
  testPrefix: string;
  data: HeatmapData | null;

  constructor(heatmapId: string, spinnerId: string, testPrefix: string) {
    this.heatmapId = heatmapId;
    this.spinnerId = spinnerId;
    this.testPrefix = testPrefix;
    this.data = null;
  }

  isLoaded() {
    return this.data !== null;
  }

  loadDataAndRender() {
    fetch('/api/get_heatmap_data', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({'test_name_prefix': this.testPrefix}),
    }).then(resp => resp.json()).then(data => {
      this.data = processData(data);
      this._render();
    });
  }

  _render() {
    const [runs, dates, tests] = this.data!;
    const height = HEIGHT_PER_TEST * tests.length;

    // Hide loading spinner
    document.getElementById(this.spinnerId)!.style.display = 'none';

    // Add the svg object to the DOM
    const svg = d3.select('#' + this.heatmapId)
      .append('svg')
        .attr('width', WIDTH + MARGIN.left + MARGIN.right)
        .attr('height', height + MARGIN.top + MARGIN.bottom)
      .append('g')
        .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);

    // Build the tooltip
    const tooltip = d3.select('#' + this.heatmapId)
      .append('div')
      .style('opacity', 0)
      .attr('class', 'tooltip')
      .style('background-color', 'white')
      .style('border', 'solid')
      .style('border-WIDTH', '2px')
      .style('border-radius', '5px')
      .style('padding', '5px')
      .style('position', 'fixed')
      .style('z-index', '1')

    // X axis
    const x = d3.scaleBand()
      .range([0, WIDTH])
      .domain(dates)
      .padding(0.05);
    svg.append('g')
      .style('font-size', 15)
      .attr('transform', 'translate(0, 0)')
      .call(d3.axisTop(x).tickSize(0))
      .selectAll('text')
      .style('text-anchor', 'end')
      .attr('dx', '0em')
      .attr('dy', '0em')
      .attr('transform', 'rotate(55)')
      .select('.domain').remove();

    // Y axis
    const y = d3.scaleBand()
      .range([height, 0])
      .domain(tests)
      .padding(0.05);
    svg.append('g')
      .style('font-size', 15)
      .call(d3.axisLeft(y).tickSize(0))
      .select('.domain').remove()

    // Event handlers for the tooltip
    const mouseover = function(this: any, event: any, run: TestRun) {
      tooltip
        .style('opacity', 1)
      d3.select(this)
        .style('stroke', 'black')
        .style('opacity', 1)
    }
    const mousemove = function(event: any, run: TestRun) {
      tooltip
        .html(`status: ${run.status}<br>date: ${run.date}`)
        .style('left', (event.x + 5) + 'px')
        .style('top', (event.y + 5) + 'px')
    }
    const mouseleave = function(this: any, event: any, run: TestRun) {
      tooltip
        .style('opacity', 0)
      d3.select(this)
        .style('stroke', 'none')
        .style('opacity', 0.8)
    }

    // Add the squares
    svg.selectAll()
      .data(runs, function(run: TestRun | undefined) {
        return `${run!.date}:${run!.test}`;
      })
      .join('rect')
        .attr('x', function(run: TestRun | undefined) {
          return x(run!.date) as number;
        })
        .attr('y', function(run: TestRun | undefined) {
          return y(run!.test) as number;
        })
        .attr('rx', 4)
        .attr('ry', 4)
        .attr('width', x.bandwidth() )
        .attr('height', y.bandwidth() )
        .style('fill', function(run: TestRun | undefined) {
          return getSquareColor(run!.status);
        })
        .style('stroke-WIDTH', 4)
        .style('stroke', 'none')
        .style('opacity', 0.8)
      .on('mouseover', mouseover)
      .on('mousemove', mousemove)
      .on('mouseleave', mouseleave)
  }
}






