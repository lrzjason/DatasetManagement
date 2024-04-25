<template>
  <v-row no-gutters>
    <v-col :cols="12" id="main-container">
      <div class="image-container">
        <v-row no-gutters align="center" justify="center">
          <v-col v-for="(img_url,index) in img_urls" :key="index">
            <span class="text-h5 image-classifier">{{ index }}</span>
            <!-- {{ image }} -->
            <v-img class="image-el" :src="img_url" @click="addToSave(img_url)">
              <v-icon v-show="selected(img_url)" class="img-status">mdi-check-circle-outline</v-icon>
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
          <!-- <v-btn class="floating-button" color="red" icon
            @click="deletePair(selectedPair)"><v-icon>mdi-delete</v-icon></v-btn>
          <v-btn class="floating-button" color="green" icon
            @click="savePair(selectedPair)"><v-icon>mdi-check</v-icon></v-btn>
          <v-btn class="floating-button" color="blue" icon
            @click="switchPair(selectedPair)"><v-icon>mdi-swap-horizontal</v-icon></v-btn>
          <v-btn class="floating-button" color="orange" icon
            @click="exportPairs()"><v-icon>mdi-export</v-icon></v-btn>
          <v-btn class="floating-button" icon @click="prevPair"><v-icon>mdi-chevron-left</v-icon></v-btn>
          <v-btn class="end-button" icon @click="nextPair"><v-icon>mdi-chevron-right</v-icon></v-btn> -->
          <v-btn class="end-button" color="green" icon ><v-icon>mdi-check</v-icon></v-btn>
        </div>
        <!-- <div class="key-control">
          <v-textarea id="key-control" width="200px" label="Key control here" rows="3"></v-textarea>
        </div> -->
      </div>
      <div class="save-prompt">
        <v-textarea v-model="save_prompt" hide-details variant="outlined" class="caption" label="save_prompt" width="400px" auto-grow rows="3"></v-textarea>
        
      </div>
      <div class="save-image-control">
        <v-textarea v-model="save_image_folder" hide-details variant="outlined" class="caption" label="save_image_folder" width="200px" auto-grow rows="1"></v-textarea>
        <v-textarea v-model="save_name" hide-details variant="outlined" class="caption ml-2" label="save_name" width="200px" auto-grow rows="1"></v-textarea>
        <v-btn class="ml-2 mt-1" color="green" icon @click="saveImage"><v-icon>mdi-content-save-all</v-icon></v-btn>
      </div>
      <div class="caption-control">
        <v-textarea v-model="caption" hide-details variant="outlined" class="caption" label="caption" width="400px" auto-grow rows="1"></v-textarea>
        <v-btn class="ml-2 mt-1" color="green" icon @click="generateImage"><v-icon>mdi-image-plus</v-icon></v-btn>
      </div>
    </v-col>
  </v-row>
</template>
<script setup>
import axios from 'axios';
import { onMounted, ref, nextTick, onUnmounted, watch } from 'vue';

const imageContainer = ref(null)

const save_prompt = ref('')
const caption = ref('')
const save_name = ref('gesture')
const updateKey = ref(0)
const cookie = ref('F:\\DatasetManagement\\flask_api\\bing_cookie.txt')
const save_image_folder = ref('F:\\ImageSet\\Dalle3')
const img_urls = ref([])
const save_img_urls = ref([])

watch(caption,(newValue)=>{
  save_prompt.value = newValue
})

function selected(url){
  if (save_img_urls.value.includes(url)){
    return true
  }
  return false
}
function addToSave(url){
  console.log('addToSave',url)
  var index = save_img_urls.value.indexOf(url);
  if (index !== -1) {
    save_img_urls.value.splice(index, 1);
  }else{
    save_img_urls.value.push(url)
  }
  console.log('save_img_urls.value',save_img_urls.value)
}

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


// const getPair = (pair,selectedIndexValue)=>{
//   selectedPair.value = pair;
//   selectedIndex.value = selectedIndexValue;
//   caption.value = pair.caption
//   goTo(`#file-${selectedIndex.value}`)
// }


// const searchPair = ()=>{
//   console.log('searchKeyword.value',searchKeyword.value)
//   console.log('pairs.value',pairs.value)

//   // find pairs by searchKeyword
//   pairs.value.map((pair,index)=>{
//     if (pair.name == searchKeyword.value){
//       selectedPair.value = pair;
//       selectedIndex.value = index;
//       caption.value = pair.caption
//       goTo(`#file-${selectedIndex.value}`)
//       return
//     }
//   })

// }

const saveImage =  debounce(() => {
  if(save_img_urls.value.length>0){
    const formData = new FormData();
    formData.append('urls',save_img_urls.value)
    formData.append('prompt',caption.value)
    formData.append('save_name',save_name.value)
    console.log('urls',save_img_urls.value)
    axios.post('http://127.0.0.1:5000/save_generation', formData)
      .then(response => {
        console.log(response)
        // img_urls.value.length = []
        // response.data.img_urls.forEach(img_url => {
        //   if (img_url.includes('http')){
        //     img_urls.value.push(img_url)
        //   }
        // });
      })
      .catch(error => {
        console.log(error);
      })
  }
}, 1000)


const generateImage =  debounce(() => {
  if(save_img_urls.value.length>0){
    if(!confirm('images haven\'t save yet. Next gen?')){
      return
    }
  }
  img_urls.value.length = 0
  save_img_urls.value.length = 0
  const formData = new FormData();
  formData.append('bing_cookie_path',cookie.value)
  formData.append('prompt',caption.value)
  console.log('generateImage',formData)
  axios.post('http://127.0.0.1:5000/generate', formData)
    .then(response => {
      img_urls.value.length = []
      response.data.img_urls.forEach(img_url => {
        if (img_url.includes('http')){
          img_urls.value.push(img_url)
        }
      });
    })
    .catch(error => {
      console.log(error);
    })
}, 1000)

// const deletePair = debounce((pair) => {
//   const formData = new FormData();
//   formData.append('image_folder', imageDir.value);
//   formData.append('caption_folder', captionDir.value);
//   formData.append('name', pair.name);
//   // async delete file
//   const index = selectedIndex.value;
//   if (index !== -1) {
//     pairs.value.splice(index, 1);
//     // roll the current index becase it is deleted
//     selectedIndex.value -= 1
//     if (index < pairs.value.length) {
//       // getPair(pairs.value[index], index);
//       nextPair()
//     }
//   }
//   axios.post('http://127.0.0.1:5000/delete_pair', formData)
//     .then(response => {
//       console.log(response.data);

//     })
//     .catch(error => {
//       console.log(error);
//     })
//   // Auto-scroll to the selected file
//   goTo(`#file-${selectedIndex.value}`)
// }, 1000)

// const goTo = (selector) => {
//   const element = document.querySelector(selector);
//   if (element) {
//     element.scrollIntoView({ behavior: 'smooth', block: 'center' });
//   }
// }

// const prevImage = () => {
//   if (selectedImageIndex.value > 0) {
//     getImage(pairs.value[selectedImageIndex.value - 1], selectedImageIndex.value - 1)
//   }
// }
// const prevPair = () => {
//   if (selectedIndex.value > 0) {
//     getPair(pairs.value[selectedIndex.value - 1], selectedIndex.value - 1)
//   }
// }

// const nextPair = () => {
//   if (selectedIndex.value < pairs.value.length - 1) {
//     if (savedPairs.value.length < pairs.value.length && pairs.value.length > 0) {
//       selectedIndex.value += 1
//       let nextPair = pairs.value[selectedIndex.value]
//       // find next unsaved pair
//       while(isSaved(nextPair.name)) {
//         selectedIndex.value += 1
//         nextPair = pairs.value[selectedIndex.value]
//       }
//     }else{
//       selectedIndex.value += 1
//     }
//     getPair(pairs.value[selectedIndex.value], selectedIndex.value);
//   }
// }

// const isSaved = (name) => {
//   // fileName = imageDir.value + "\\" + fileName.split('.')[0] + '.txt';
//   return savedPairs.value.includes(name);
// }

// const getImageOrThumbnail = (pair) =>{
//   console.log(pair)
//   if(pair.thumbnails){
//     if(Object.keys(pair.thumbnails).length>0){
//       return pair.thumbnails
//     }
//   }
//   return pair.images
// }

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
    // switch (event.key) {
    //   case 'ArrowLeft': // Left arrow
    //     prevPair();
    //     break;
    //   case 'ArrowRight': // Right arrow
    //     nextPair();
    //     break;
    //   case 'PageDown': // F1
    //     switchPair(selectedPair.value);
    //     event.preventDefault();
    //     break;
    //   case 'End': // F1
    //     savePair(selectedPair.value);
    //     event.preventDefault();
    //     break;
    //   case 'Delete': // F2
    //     deletePair(selectedPair.value);
    //     event.preventDefault();
    //     break;
    // }
  }
};

onMounted(() => {
  // listFiles();
  // listPairs();
  // let container = document.querySelector('#key-control')
  // console.log('container',container)
  // container.addEventListener('keydown', handleKeyboardControls);
  document.addEventListener('keydown', handleKeyboardControls);
})

onUnmounted(() => {
  unbindKeyboardControls();
});

</script>

<style scoped>
.img-status{
  position: absolute;
  top:0;
  left:0;
  font-size: 50px;
  color: rgb(0, 185, 0);
  background-color: rgba(255, 255, 255, 0.5);
}
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
  color:white;
  background-color: rgba(0, 0, 0, 0.5);
  width: 600px;
  position: absolute;
  bottom: 50px;
  left: 20vw;
  padding: 10px;
  display: inline-flex;
}
.save-prompt{
  color:white;
  background-color: rgba(0, 0, 0, 0.5);
  width: 600px;
  position: absolute;
  bottom: 150px;
  left: 50vw;
  padding: 10px;
  display: inline-flex;
}
.save-image-control{
  color:white;
  background-color: rgba(0, 0, 0, 0.5);
  width: 600px;
  position: absolute;
  bottom: 50px;
  left: 50vw;
  padding: 10px;
  display: inline-flex;
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
