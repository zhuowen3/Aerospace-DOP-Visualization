<template>
  <div>
    <div id="mapid" style="height:512px; width:512px"></div>
  </div>
</template>

<script>
import L from 'leaflet';
import 'leaflet-contour';
import "leaflet/dist/leaflet.css";

export default {
  data() {
    return {
      //setting up 2D array where x = longitude, y = latitude, and z = dop value
      data: {
        x: [],
        y: [],
        z: []
      },
      //min & max set beyond real coordinates to compensate for leaflet-contour limitation where surrounding null values are required
      latMin: -91.44,
      latMax: 91.44,
      lngMin: -181.44,
      lngMax: 181.44,
      //One DOP value for every 0.72 coordinate point
      interval: 0.72,
      //color gradient for the contour plot
      colors: [
        { color: "#00008f", point: 0 },
        { color: "#0000ef", point: 0.11111111111 },
        { color: "#005fff", point: 0.22222222222 },
        { color: "#00cfff", point: 0.33333333333 },
        { color: "#4fffaf", point: 0.44444444444 },
        { color: "#bfff3f", point: 0.55555555556 },
        { color: "#ffcf00", point: 0.66666666667 },
        { color: "#ff5f00", point: 0.77777777778 },
        { color: "#ef0000", point: 0.88888888889 },
        { color: "#7f0000", point: 0.1 },

      ]
    };
  },
  async mounted() {
    //initialize data arrays
    await this.generateData();
    //populate z array with DOP data
    await this.updateZValues("/dop_output_3.txt");
    //show world map
    this.initializeMap();
  },
  methods: {
    async parseTextFile(filePath) {
      const response = await fetch(filePath);
      const text = await response.text();
      const rows = text.split("\n").filter((line) => line.trim() !== "");
      const parsedData = [];
      rows.forEach((row) => {
        const [lat, lng, gdop, pdop, hdop, vdop, tdop, numInView] = row.split(/\s+/);
        parsedData.push({
          latitude: parseFloat(lat),
          longitude: parseFloat(lng),
          gdop: parseFloat(gdop),
          pdop: parseFloat(pdop),
          hdop: parseFloat(hdop),
          vdop: parseFloat(vdop),
          tdop: parseFloat(tdop),
          numInView: parseInt(numInView)
        });
      });
      return parsedData;

    },

    //navigates through the 2d array to assign z-values
    getIndices(lat, lng, latMax, lngMin, interval) {
      const i = Math.round((latMax - lat) / interval);
      const j = Math.round((lng - lngMin) / interval);
      return { i, j };
    },

    //assigns longitude, latitude, and null values into x, y, and z respectively
    generateData() {
      const numRows = Math.floor((this.latMax - this.latMin) / this.interval) + 1;
      const numCols = Math.floor((this.lngMax - this.lngMin) / this.interval) + 1;

      for (let i = 0; i < numRows; i++) {
        const rowX = [];
        const rowY = [];
        const rowZ = [];

        const lat = this.latMax - i * this.interval;

        for (let j = 0; j < numCols; j++) {
          const lng = this.lngMin + j * this.interval;

          rowX.push(lng);
          rowY.push(lat);
          rowZ.push(null);

        }

        this.data.x.push(rowX);
        this.data.y.push(rowY);
        this.data.z.push(rowZ);
      }
      
      console.log(this.data);

    },

    //Changes z-values based on provided DOP data
    async updateZValues(filePath) {
      const parsedData = await this.parseTextFile(filePath);
      parsedData.forEach(({ latitude, longitude, pdop }) => {
        const { i, j } = this.getIndices(latitude, longitude, this.latMax, this.lngMin, this.interval);
        if (i >= 0 && i < this.data.z.length && j >= 0 && j < this.data.z[i].length) {
          this.data.z[i][j] = pdop;
        }
      });

    },

    //copied from leaflet-contour; mumbo jumbo about contour color calculation
    getColor(value, min, max, colors) {
      function hex(c) {
        var s = "0123456789abcdef";
        var i = parseInt(c, 10);
        if (i === 0 || isNaN(c)) return "00";
        i = Math.round(Math.min(Math.max(0, i), 255));
        return s.charAt((i - (i % 16)) / 16) + s.charAt(i % 16);
      }
      function trim(s) {
        return s.charAt(0) === "#" ? s.substring(1, 7) : s;
      }
      function convertToRGB(hex) {
        var color = [];
        color[0] = parseInt(trim(hex).substring(0, 2), 16);
        color[1] = parseInt(trim(hex).substring(2, 4), 16);
        color[2] = parseInt(trim(hex).substring(4, 6), 16);
        return color;
      }
      function convertToHex(rgb) {
        return hex(rgb[0]) + hex(rgb[1]) + hex(rgb[2]);
      }

      if (value === null || isNaN(value)) {
        return "#ffffff";
      }
      if (value > max) {
        return colors[colors.length - 1].color;
      }
      if (value < min) {
        return colors[0].color;
      }
      var loc = (value - min) / (max - min);
      if (loc < 0 || loc > 1) {
        return "#fff";
      } else {
        var index = 0;
        for (var i = 0; i < colors.length - 1; i++) {
          if (loc >= colors[i].point && loc <= colors[i + 1].point) {
            index = i;
          }
        }
        var color1 = convertToRGB(colors[index].color);
        var color2 = convertToRGB(colors[index + 1].color);

        var f =
          (loc - colors[index].point) /
          (colors[index + 1].point - colors[index].point);
        var rgb = [
          color1[0] + (color2[0] - color1[0]) * f,
          color1[1] + (color2[1] - color1[1]) * f,
          color1[2] + (color2[2] - color1[2]) * f,
        ];

        return `#${convertToHex(rgb)}`;
      }
    },
    //creates the Leaflet Map
    initializeMap() {
      var map = L.map("mapid", {
        worldCopyJump: true,
        maxBounds: [
          [-90, -180],
          [90, 180]
        ],
        minZoom: 1,
        maxBoundsViscosity: 1,
      }).setView([0, 0], 1);

      L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png}",
        {
          maxZoom: 7,
          ext: 'png',
          tileSize: 512,
          zoomOffset: -1,
        }
      ).addTo(map);
      //creates outline of country borders on the world map
      fetch('/worldmap.json')
        .then(response => response.json())
        .then(data => {
          const geojsonLayer = L.geoJSON(data, {
            style: function () {
              return {
                color: "white",
                weight: 1,
                fillOpacity: 0,
                interactive: false
              };
            }
          });
          geojsonLayer.addTo(map);
        })
      //
      L.contour(this.data, {
        //adjust number of contours on the map; adjust based on data variance
        thresholds: 3, 

        style: (feature) => {
          return {
            color: this.getColor(feature.geometry.value, 1, 10, this.colors),
            opacity: 0,
            fillOpacity: 1,
          };
        },
        onEachFeature: this.onEachContour(), //shows the mean DOP value of a contour layer
      }).addTo(map);

      //creates the color gradient legend
      const legend = L.control({ position: 'bottomright' });

      legend.onAdd = function () {
        const div = L.DomUtil.create('div', 'info legend');
        const grades = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];  // Adjust depending on the threshold
        const colors = [
          "#7f0000", "#ef0000", "#ff5f00", "#ffcf00", "#bfff3f", "#4fffaf", "#00cfff", "#005fff", "#0000ef", "#00008f"
        ];

        let gradient = 'linear-gradient(to top,';
        for (let i = colors.length - 1; i >= 0; i--) {
          gradient += ` ${colors[i]},`;
        }
        gradient = gradient.slice(0, -1) + ')';

        div.innerHTML = `
    <div style="background: ${gradient}; height: 450px; width: 15px; position: relative;">
      ${grades.map((value, i) => `<div style="position: absolute; bottom: ${(i / (grades.length - 1)) * 100}%; color: white; font-size: 10px; right: 25px;">${value}</div>`).join('')}
    </div>
  `;
        return div;
      };

      legend.addTo(map);

    },

    //Leaflet popup that should show the value (DOP) of a clicked contour
    onEachContour() {
      return function onEachFeature(feature, layer) {
        //eslint-disable-next-line
        let roundedDOP = Math.ceil(feature.value);
        layer.bindPopup(
          `<table>
              <tbody>
                <tr><td>PDOP: ${feature.value}</td></tr>
              </tbody>
            </table>`
        );
      };
    }
  }
};
</script>

<style>
#mapid {
  background-color: black;
}
</style>