<template>
  <!-- <v-row>
    <v-col>
      <v-text-field v-model="imageDir" label="Target Dir">
      </v-text-field>
    </v-col>
  </v-row> -->
  <v-row no-gutters>
    <v-col cols="3" v-show="!hideList">
      <h1>List of Files</h1>
      <v-card class="scrollable-list">
        <v-list>
          <v-list-item :class="{ 'selected-file': selectedPair.name === pair.name }" v-for="(pair, index) in pairs" :key="pair.name"
            @click="getPair(pair, index)" :id="`file-${index}`">
            <template v-slot:prepend>
              <v-icon v-if="isSaved(pair.name)" color="green">mdi-check</v-icon>
            </template>

            <v-list-item-title>{{ pair.name }}</v-list-item-title>
          </v-list-item>
        </v-list>
      </v-card>
      <!-- <v-textarea v-model="caption" label="Enter text here" rows="10"></v-textarea> -->
      <v-text-field v-model="searchKeyword" label="search" rows="10"></v-text-field>
      <v-btn class="floating-button" color="blue" icon
        @click="searchPair"><v-icon>mdi-text-search</v-icon></v-btn>
    </v-col>
    <v-col :cols="hideList?12:9" id="main-container">
      <div>Selected Pair: {{ selectedPair.name }} {{ savedPairs.length }} / {{ pairs.length }}</div>
      <!-- <div class="controls-wrapper">
        <div class="image-controls">
          <v-btn icon @click="prevImage"><v-icon>mdi-chevron-left</v-icon></v-btn>
          <v-btn icon @click="nextImage"><v-icon>mdi-chevron-right</v-icon></v-btn>
        </div>
      </div> -->
      <div class="image-container">
        <v-row no-gutters align="center" justify="center">
          <v-col v-for="(image,index) in getImageOrThumbnail(selectedPair)" :key="index">
            <span class="text-h5 image-classifier">{{ index }}</span>
            <!-- {{ image }} -->
            <v-img class="image-el" :src="`http://127.0.0.1:5000/file/${image}?${updateKey}`">
              <template v-slot:placeholder>
                <v-row class="fill-height ma-0" align="center" justify="center">
                  <v-progress-circular indeterminate color="grey-lighten-5"></v-progress-circular>
                </v-row>
              </template>
            </v-img>
          </v-col>
          <!-- <v-col>
            <v-img :src="imageSrc">
              <template v-slot:placeholder>
                <v-row class="fill-height ma-0" align="center" justify="center">
                  <v-progress-circular indeterminate color="grey-lighten-5"></v-progress-circular>
                </v-row>
              </template>
            </v-img>
          </v-col> -->
        </v-row>
        <div class="image-buttons">
          <v-btn class="floating-button" color="red" icon
            @click="deletePair(selectedPair)"><v-icon>mdi-delete</v-icon></v-btn>
          <v-btn class="floating-button" color="green" icon
            @click="savePair(selectedPair)"><v-icon>mdi-check</v-icon></v-btn>
          <v-btn class="floating-button" color="blue" icon
            @click="switchPair(selectedPair)"><v-icon>mdi-swap-horizontal</v-icon></v-btn>
          <v-btn class="floating-button" color="orange" icon
            @click="exportPairs()"><v-icon>mdi-export</v-icon></v-btn>
          <v-btn class="floating-button" icon @click="prevPair"><v-icon>mdi-chevron-left</v-icon></v-btn>
          <v-btn class="end-button" icon @click="nextPair"><v-icon>mdi-chevron-right</v-icon></v-btn>
        </div>
        <!-- <div class="key-control">
          <v-textarea id="key-control" width="200px" label="Key control here" rows="3"></v-textarea>
        </div> -->
      </div>
      <div class="caption-control">
        <v-textarea v-model="caption" hide-details variant="outlined" class="caption" label="caption" width="400px" auto-grow rows="1"></v-textarea>
      </div>
    </v-col>
  </v-row>
</template>

<style scoped>
.image-el{
  max-height: calc(100vh - 110px);
  margin-top: 75px;
}
.caption textarea {
  text-align: center;
  font-size: 1.2em;
  color: white;
}
.image-classifier{
  color: white;
  position: absolute;
  z-index: 999;
}
.controls-wrapper {
  position: relative;
  top: calc(100vh/2 + 100px - 50px);
  z-index: 9999;
}

.image-container {
  display: flex;
  align-items: center;
}

.scrollable-list {
  max-height: calc(100vh - 450px);
  overflow-y: auto;
}

.image-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.caption-control{
  background-color: rgba(0, 0, 0, 0.5);
  width: 600px;
  position: absolute;
  bottom: 0px;
  left: 35vw;
  padding: 10px;
}

.key-control{
  position: absolute;
  top: 70px;
  right: 50px;
  display: flex;
  flex-direction: row;
  align-items: flex-end;
  padding: 10px;
}

.image-buttons {
  position: absolute;
  top: 0;
  right: 0;
  display: flex;
  flex-direction: row;
  align-items: flex-end;
  padding: 10px;
}

.floating-button {
  /* margin-bottom: 30px; */
  margin-right: 30px;
}

.end-button {
  margin-right: 50px;
}

.selected-file {
  background-color: #ddd;
}
</style>

<script setup>
import axios from 'axios';
import { onMounted, ref, nextTick, onUnmounted, watch } from 'vue';

const searchKeyword = ref('')

const hideList = ref(false)

const imageContainer = ref(null)

const caption = ref('')
const updateKey = ref(0)

// const selectedImage = ref('')
// const selectedImageIndex = ref(0)
// const selectedPairIndex = ref(0)

// const files = ref([])
const pairs = ref([])
// const savedFiles = ref([])
const savedPairs = ref([])
const imageDir = ref('F:\\ImageSet\\Pickscore_train_10k\\images')
const captionDir = ref('F:\\ImageSet\\Pickscore_train_10k\\captions')

// watch imageDir change, list files
watch(imageDir, (newValue, oldValue) => {
  listPairs();
})

const selectedImageName = ref('')
const selectedPair = ref({name:''})
const selectedIndex = ref(1)

const debounce = (func, delay) => {
  let timeoutId;
  return (...args) => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
      func.apply(null, args);
    }, delay);
  };
};

const imageSrc = ref('')

const CACHE_SIZE = 20;
const imageCacheList = [];


const cacheImageData = (imagePath, imageData) => {
  if (imageCacheList.length >= CACHE_SIZE) {
    // Remove the oldest cached image
    imageCacheList.shift();
  }
  imageCacheList.push({ path: imagePath, data: imageData });
};

const cachingFiles = ref([])

const getPair = (pair,selectedIndexValue)=>{
  selectedPair.value = pair;
  selectedIndex.value = selectedIndexValue;
  caption.value = pair.caption
  goTo(`#file-${selectedIndex.value}`)
}


const searchPair = ()=>{
  console.log('searchKeyword.value',searchKeyword.value)
  console.log('pairs.value',pairs.value)

  // find pairs by searchKeyword
  pairs.value.map((pair,index)=>{
    if (pair.name == searchKeyword.value){
      selectedPair.value = pair;
      selectedIndex.value = index;
      caption.value = pair.caption
      goTo(`#file-${selectedIndex.value}`)
      return
    }
  })

}

const switchPair = debounce((pair) => {
  console.log('switch pair', pair.name)
  const formData = new FormData();
  formData.append('switch_from', pair.images[Object.keys(pair.images)[0]]);
  formData.append('switch_to', pair.images[Object.keys(pair.images)[1]]);
  axios.post('http://127.0.0.1:5000/switch_pair', formData)
    .then(response => {
      console.log(response.data);
      updateKey.value += 1
    })
    .catch(error => {
      console.log(error);
    });
},1000)

const savePair = debounce((pair) => {
  // Auto-scroll to the selected file
  goTo(`#file-${selectedIndex.value + 1}`)
  console.log('save pair', pair.name)
  const formData = new FormData();
  formData.append('name', pair.name);
  formData.append('file_name', captionDir.value + '\\' + pair.name + '.txt');
  formData.append('content', caption.value);
  // async save file
  nextPair();

  axios.post('http://127.0.0.1:5000/save_pair', formData)
    .then(response => {
      console.log(response.data);
      savedPairs.value.push(pair.name)
      // savedFiles.value = [...response.data.saved_files];
      // savedPairs.value = [...response.data.saved_pairs];
    })
    .catch(error => {
      console.log(error);
    });
}, 1000)

const deletePair = debounce((pair) => {
  const formData = new FormData();
  formData.append('image_folder', imageDir.value);
  formData.append('caption_folder', captionDir.value);
  formData.append('name', pair.name);
  // async delete file
  const index = selectedIndex.value;
  if (index !== -1) {
    pairs.value.splice(index, 1);
    // roll the current index becase it is deleted
    selectedIndex.value -= 1
    if (index < pairs.value.length) {
      // getPair(pairs.value[index], index);
      nextPair()
    }
  }
  axios.post('http://127.0.0.1:5000/delete_pair', formData)
    .then(response => {
      console.log(response.data);

    })
    .catch(error => {
      console.log(error);
    })
  // Auto-scroll to the selected file
  goTo(`#file-${selectedIndex.value}`)
}, 1000)

const goTo = (selector) => {
  const element = document.querySelector(selector);
  if (element) {
    element.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

// const prevImage = () => {
//   if (selectedImageIndex.value > 0) {
//     getImage(pairs.value[selectedImageIndex.value - 1], selectedImageIndex.value - 1)
//   }
// }
const prevPair = () => {
  if (selectedIndex.value > 0) {
    getPair(pairs.value[selectedIndex.value - 1], selectedIndex.value - 1)
  }
}

const nextPair = () => {
  if (selectedIndex.value < pairs.value.length - 1) {
    if (savedPairs.value.length < pairs.value.length && pairs.value.length > 0) {
      selectedIndex.value += 1
      let nextPair = pairs.value[selectedIndex.value]
      // find next unsaved pair
      while(isSaved(nextPair.name)) {
        selectedIndex.value += 1
        nextPair = pairs.value[selectedIndex.value]
      }
    }else{
      selectedIndex.value += 1
    }
    getPair(pairs.value[selectedIndex.value], selectedIndex.value);
  }
}

const isSaved = (name) => {
  // fileName = imageDir.value + "\\" + fileName.split('.')[0] + '.txt';
  return savedPairs.value.includes(name);
}

const exportPairs = debounce(() => {
  const formData = new FormData();
  formData.append('image_folder', imageDir.value);
  formData.append('caption_folder', captionDir.value);
  axios.post('http://127.0.0.1:5000/export_pairs', formData)
    .then(response => {
      console.log('response',response)
    })
    .catch(error => {
      console.log(error);
    });
}, 1000)


const listPairs = () => {
  const formData = new FormData();
  formData.append('image_folder', imageDir.value);
  formData.append('caption_folder', captionDir.value);
  axios.post('http://127.0.0.1:5000/list_pairs', formData)
    .then(response => {

      console.log('response',response)
      pairs.value = response.data.pairs
      nextTick(() => {
        savedPairs.value = [...response.data.saved_pairs];
        if (pairs.value.length > 0) {
          let unsavedIndex = -1
          for (let index = 0; index < pairs.value.length; index++) {
            const record = pairs.value[index];
            if (!isSaved(record.name)){
              unsavedIndex = index
              break
            }
          }
          // const unsavedIndex = pairs.value.forEach(record=>record.name).findIndex(pair => !isSaved(pair.name));
          if (unsavedIndex !== -1) {
            getPair(pairs.value[unsavedIndex], unsavedIndex);
          } else {
            getPair(pairs.value[0], 0);
          }
        }
      })
    })
    .catch(error => {
      console.log(error);
    });
}

const getImageOrThumbnail = (pair) =>{
  console.log(pair)
  if(pair.thumbnails){
    if(Object.keys(pair.thumbnails).length>0){
      return pair.thumbnails
    }
  }
  return pair.images
}

const unbindKeyboardControls = () => {
  // let container = document.querySelector('#key-control')
  // console.log('container',container)
  // container.removeEventListener('keydown', handleKeyboardControls);
  document.removeEventListener('keydown', handleKeyboardControls);
};

const handleKeyboardControls = (event) => {
  console.log(event.key);
  const focusedElement = document.activeElement;
  if (focusedElement.tagName !== 'TEXTAREA') {
    switch (event.key) {
      case 'ArrowLeft': // Left arrow
        prevPair();
        break;
      case 'ArrowRight': // Right arrow
        nextPair();
        break;
      case 'PageDown': // F1
        switchPair(selectedPair.value);
        event.preventDefault();
        break;
      case 'End': // F1
        savePair(selectedPair.value);
        event.preventDefault();
        break;
      case 'Delete': // F2
        deletePair(selectedPair.value);
        event.preventDefault();
        break;
    }
  }
};

onMounted(() => {
  // listFiles();
  listPairs();
  // let container = document.querySelector('#key-control')
  // console.log('container',container)
  // container.addEventListener('keydown', handleKeyboardControls);
  document.addEventListener('keydown', handleKeyboardControls);
})

onUnmounted(() => {
  unbindKeyboardControls();
});

</script>