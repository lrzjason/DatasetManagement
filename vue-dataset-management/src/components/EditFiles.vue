<template>
  <div>
    <input type="file" @change="onFileSelected">
    <textarea v-model="fileContents"></textarea>
    <button @click="saveFile">Save</button>
  </div>
</template>

<script>
export default {
  data() {
    return {
      file: null,
      fileContents: ''
    }
  },
  methods: {
    onFileSelected(event) {
      this.file = event.target.files[0]
      const reader = new FileReader()
      reader.onload = () => {
        this.fileContents = reader.result
      }
      reader.readAsText(this.file)
    },
    saveFile() {
      window.requestFileSystem(window.TEMPORARY, 1024 * 1024, (fs) => {
        fs.root.getFile(this.file.name, { create: true }, (fileEntry) => {
          fileEntry.createWriter((fileWriter) => {
            const blob = new Blob([this.fileContents], { type: 'text/plain' })
            fileWriter.write(blob)
          })
        })
      })
    }
  }
}
</script>