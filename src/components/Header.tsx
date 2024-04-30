import { chakra, Heading, Wrap, Box, Container, Text, Link as ChakraLink, useColorModeValue } from '@chakra-ui/react'
import NextLink from "next/link"
import { SimpleGrid } from '@chakra-ui/react'

import { title, authors, author_tags } from 'data'


export const Title = () => (
  <Heading fontSize={{ base: '3xl', md: '4xl' }} pt={{ base: '2.5vh', md: '5vh' }} pb={{ base: '1rem', md: '1rem' }} fontWeight="700">{title}</Heading>
)
