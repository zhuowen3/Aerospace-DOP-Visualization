<template>
  <div>
    <VueDatePicker v-model="date"></VueDatePicker>
    <p>Selected Date: {{ date }}</p>
    <div id="mapid" style="height:512px; width:545px"></div>
  </div>
</template>

<script setup>
import VueDatePicker from "@vuepic/vue-datepicker";
import "@vuepic/vue-datepicker/dist/main.css";

</script>

<script>
import L from "leaflet";
import "leaflet-contour";
import "leaflet/dist/leaflet.css";

export default {
  data() {
    return {
      date: new Date(),
      data: {
        x: [],
        y: [],
        z: []
      },
      map: null,
      latMin: -91.44,
      latMax: 91.44,
      lngMin: -181.44,
      lngMax: 181.44,
      interval: 0.72,
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
        { color: "#ef0000", point: 0.99999999999 },
        { color: "#7f0000", point: 1.0 },
      ]
    };
  },
  async mounted() {
    this.generateData();
    await this.fetchDataFromFastAPI();
    this.$nextTick(() => {
        this.initializeMap();
    });
  },
  watch: {
    async date(newDate) {
      console.log("Data changed:", newDate);
      console.log(this.data);
      await this.fetchDataFromFastAPI();
      this.restartMap();
    }
  },
  methods: {
    async fetchDataFromFastAPI() {
      try {
        const response = await fetch("http://localhost:8000/data/dop");
        const jsonResponse = await response.json();
        const dopData = jsonResponse.results;

        // Clear existing data
        this.data.z.forEach(row => row.fill(null));

        dopData.forEach(({time, Latitude, Longitude, PDOP }) => {
          const lat = parseFloat(Latitude);
          const lng = parseFloat(Longitude);
          const pdop = parseFloat(PDOP);

          // Parse the time from InfluxDB and compare it with the selected date
          const timestamp = new Date(time); // Assuming the time is in ISO 8601 format
          const selectedDate = new Date(this.date); // Assuming the date is selected in the date picker
      
          // Ensure the comparison checks if the time is close to the selected date
          const timeDiff = Math.abs(timestamp - selectedDate);
          const threshold = 60 * 60 * 1000; // 1 hour in milliseconds

          if (timeDiff <= threshold) { // Check if the time is within the threshold (e.g., 1 hour)
            const { i, j } = this.getIndices(lat, lng, this.latMax, this.lngMin, this.interval);

            if (i >= 0 && i < this.data.z.length && j >= 0 && j < this.data.z[i].length) {
              this.data.z[i][j] = Math.floor(pdop);
            }
          }
        });

      } catch (error) {
        console.error("Error fetching data from FastAPI:", error);
      }
    },
    getIndices(lat, lng, latMax, lngMin, interval) {
      const i = Math.round((latMax - lat) / interval);
      const j = Math.round((lng - lngMin) / interval);
      return { i, j };
    },
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


    },
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
    initializeMap() {
      this.map = L.map("mapid", {
        worldCopyJump: true,
        maxBounds: [[-90, -180], [90, 205]],
        minZoom: 1,
        maxBoundsViscosity: 1,
        preferCanvas: true,
        inertia: false
      }).setView([0, 0], 1);

      L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png}", {
        maxZoom: 5,
        ext: "png",
        tileSize: 512,
        zoomOffset: -1
      }).addTo(this.map);

      fetch("/worldmap.json")
        .then(response => response.json())
        .then(data => {
          const geojsonLayer = L.geoJSON(data, {
            style: () => ({
              color: "white",
              weight: 1,
              fillOpacity: 0,
              interactive: false
            })
          });
          geojsonLayer.addTo(this.map);
        });

      L.contour(this.data, {
        thresholds: 10,
        style: feature => ({
          color: this.getColor(feature.geometry.value, 1, 10, this.colors),
          opacity: 0,
          fillOpacity: 1
        }),
        onEachFeature: this.onEachContour()
      }).addTo(this.map);
      
      const legend = L.control({ position: "bottomright" });

      legend.onAdd = function () {
        const div = L.DomUtil.create("div", "info legend");
        const grades = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        const colors = ["#7f0000", "#ef0000", "#ff5f00", "#ffcf00", "#bfff3f", "#4fffaf", "#00cfff", "#005fff", "#0000ef", "#00008f"];

        let gradient = "linear-gradient(to top,";
        for (let i = colors.length - 1; i >= 0; i--) {
          gradient += ` ${colors[i]},`;
        }
        gradient = gradient.slice(0, -1) + ")";

        div.innerHTML = `
          <div style="background: ${gradient}; height: 460px; width: 20px; position: relative;">
            ${grades.map((value, i) => `<div style="position: absolute; bottom: ${(i / (grades.length - 1)) * 100}%; color: white; font-size: 10px; right: 25px;">${value}</div>`).join("")}
          </div>
        `;
        return div;
      };

      legend.addTo(this.map);
    },
    restartMap(){
      if (this.map) {
        this.map.stop();
        this.map.eachLayer(layer => this.map.removeLayer(layer)); // Remove all layers
        this.map.off();
        this.map.remove(); // Properly destroy the map instance
        this.map = null; // Clear reference // Properly remove the existing map instance
      }

      this.$nextTick(() => {
        if (!this.map) {
          this.initializeMap();
        }
      });
      
    },
    onEachContour() {
      return function (feature, layer) {
        // let roundedDOP = Math.round(feature.value);
        layer.bindPopup(`<table><tbody><tr><td>PDOP: ${feature.value}</td></tr></tbody></table>`);
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
