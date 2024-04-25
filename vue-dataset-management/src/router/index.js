import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import PairsView from '../views/PairsView.vue'
import PrepareView from '../views/PrepareView.vue'
import GenerateImageView from '../views/GenerateImageView.vue'
import DefaultLayout from '@/layouts/DefaultLayout.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
      meta: { layout: DefaultLayout }
    },
    {
      path: '/pairs',
      name: 'pairs',
      component: PairsView,
      meta: { layout: DefaultLayout }
    },
    {
      path: '/prepare',
      name: 'prepare',
      component: PrepareView,
      meta: { layout: DefaultLayout }
    },
    {
      path: '/generateImage',
      name: 'generateImageView',
      component: GenerateImageView,
      meta: { layout: DefaultLayout }
    },
    {
      path: '/about',
      name: 'about',
      // route level code-splitting
      // this generates a separate chunk (About.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import('../views/AboutView.vue'),
      // component: () => import('../views/AboutView.vue')
     meta: { layout: DefaultLayout }
      
    }
  ]
})

export default router
