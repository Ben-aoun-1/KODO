import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import * as d3 from 'd3'
import { ZoomIn, ZoomOut, Maximize2, RefreshCw, Search } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useAppStore } from '@/store'
import type { CallGraph, GraphNode, GraphEdge } from '@/types'

interface SimulationNode extends GraphNode, d3.SimulationNodeDatum {}

interface SimulationLink extends d3.SimulationLinkDatum<SimulationNode> {
  type: GraphEdge['type']
}

const nodeColors: Record<string, string> = {
  function: '#3b82f6',
  method: '#8b5cf6',
  class: '#f59e0b',
  module: '#10b981',
}

const edgeColors: Record<string, string> = {
  calls: '#6b7280',
  uses: '#9ca3af',
  imports: '#d1d5db',
  inherits: '#f97316',
}

export function Graph() {
  const navigate = useNavigate()
  const { currentRepo, selectedEntity, callGraph, isLoadingGraph, loadCallGraph } =
    useAppStore()
  const svgRef = useRef<SVGSVGElement>(null)
  const [depth, setDepth] = useState('2')
  const [searchQuery, setSearchQuery] = useState('')
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)

  useEffect(() => {
    if (selectedEntity && currentRepo) {
      loadCallGraph(selectedEntity.id, parseInt(depth))
    }
  }, [selectedEntity, currentRepo, depth, loadCallGraph])

  useEffect(() => {
    if (!callGraph || !svgRef.current) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = svgRef.current.clientWidth
    const height = svgRef.current.clientHeight

    // Create zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform)
      })

    svg.call(zoom)

    const container = svg.append('g')

    // Create arrow markers for directed edges
    const defs = svg.append('defs')

    Object.entries(edgeColors).forEach(([type, color]) => {
      defs
        .append('marker')
        .attr('id', `arrow-${type}`)
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 20)
        .attr('refY', 0)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', color)
    })

    // Prepare data for simulation
    const nodes: SimulationNode[] = callGraph.nodes.map((n) => ({ ...n }))
    const links: SimulationLink[] = callGraph.edges.map((e) => ({
      source: e.source,
      target: e.target,
      type: e.type,
    }))

    // Create force simulation
    const simulation = d3
      .forceSimulation(nodes)
      .force(
        'link',
        d3
          .forceLink<SimulationNode, SimulationLink>(links)
          .id((d) => d.id)
          .distance(100)
      )
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(40))

    // Draw links
    const link = container
      .append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', (d) => edgeColors[d.type])
      .attr('stroke-width', 2)
      .attr('marker-end', (d) => `url(#arrow-${d.type})`)

    // Draw nodes
    const node = container
      .append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .call(
        d3
          .drag<SVGGElement, SimulationNode>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart()
            d.fx = d.x
            d.fy = d.y
          })
          .on('drag', (event, d) => {
            d.fx = event.x
            d.fy = event.y
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0)
            d.fx = null
            d.fy = null
          })
      )

    // Node circles
    node
      .append('circle')
      .attr('r', (d) => (d.id === callGraph.rootId ? 20 : 15))
      .attr('fill', (d) => nodeColors[d.type] || '#6b7280')
      .attr('stroke', (d) => (d.id === callGraph.rootId ? '#fff' : 'none'))
      .attr('stroke-width', 2)

    // Node labels
    node
      .append('text')
      .text((d) => d.name)
      .attr('text-anchor', 'middle')
      .attr('dy', 30)
      .attr('font-size', 12)
      .attr('fill', 'currentColor')

    // Hover effects
    node
      .on('mouseover', (_, d) => setHoveredNode(d))
      .on('mouseout', () => setHoveredNode(null))

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d) => (d.source as SimulationNode).x!)
        .attr('y1', (d) => (d.source as SimulationNode).y!)
        .attr('x2', (d) => (d.target as SimulationNode).x!)
        .attr('y2', (d) => (d.target as SimulationNode).y!)

      node.attr('transform', (d) => `translate(${d.x},${d.y})`)
    })

    // Center the graph
    svg.call(zoom.transform, d3.zoomIdentity.translate(0, 0).scale(0.8))

    return () => {
      simulation.stop()
    }
  }, [callGraph])

  const handleZoom = (factor: number) => {
    if (!svgRef.current) return
    const svg = d3.select(svgRef.current)
    const zoom = d3.zoom<SVGSVGElement, unknown>()
    svg.transition().duration(300).call(zoom.scaleBy, factor)
  }

  const handleReset = () => {
    if (!svgRef.current) return
    const svg = d3.select(svgRef.current)
    const zoom = d3.zoom<SVGSVGElement, unknown>()
    svg
      .transition()
      .duration(500)
      .call(zoom.transform, d3.zoomIdentity.translate(0, 0).scale(0.8))
  }

  if (!currentRepo) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <p className="text-muted-foreground">Select a repository first</p>
        <Button onClick={() => navigate('/repositories')}>
          Go to Repositories
        </Button>
      </div>
    )
  }

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Call Graph</h1>
          <p className="text-muted-foreground mt-1">
            Visualize function call relationships
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={depth} onValueChange={setDepth}>
            <SelectTrigger className="w-32">
              <SelectValue placeholder="Depth" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1">Depth 1</SelectItem>
              <SelectItem value="2">Depth 2</SelectItem>
              <SelectItem value="3">Depth 3</SelectItem>
              <SelectItem value="4">Depth 4</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex-1 grid grid-cols-12 gap-4">
        {/* Controls */}
        <Card className="col-span-3">
          <CardHeader className="py-3">
            <CardTitle className="text-sm">Controls</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-2">
              <Button variant="outline" size="icon" onClick={() => handleZoom(1.5)}>
                <ZoomIn className="h-4 w-4" />
              </Button>
              <Button variant="outline" size="icon" onClick={() => handleZoom(0.67)}>
                <ZoomOut className="h-4 w-4" />
              </Button>
              <Button variant="outline" size="icon" onClick={handleReset}>
                <Maximize2 className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={() => selectedEntity && loadCallGraph(selectedEntity.id, parseInt(depth))}
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>

            <div>
              <p className="text-sm font-medium mb-2">Legend</p>
              <div className="space-y-2">
                {Object.entries(nodeColors).map(([type, color]) => (
                  <div key={type} className="flex items-center gap-2 text-sm">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: color }}
                    />
                    <span className="capitalize">{type}</span>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <p className="text-sm font-medium mb-2">Edge Types</p>
              <div className="space-y-2">
                {Object.entries(edgeColors).map(([type, color]) => (
                  <div key={type} className="flex items-center gap-2 text-sm">
                    <div
                      className="w-8 h-0.5"
                      style={{ backgroundColor: color }}
                    />
                    <span className="capitalize">{type}</span>
                  </div>
                ))}
              </div>
            </div>

            {hoveredNode && (
              <div className="border-t pt-4">
                <p className="text-sm font-medium mb-2">Selected Node</p>
                <div className="text-sm space-y-1">
                  <p>
                    <span className="text-muted-foreground">Name:</span>{' '}
                    {hoveredNode.name}
                  </p>
                  <p>
                    <span className="text-muted-foreground">Type:</span>{' '}
                    {hoveredNode.type}
                  </p>
                  <p className="truncate">
                    <span className="text-muted-foreground">File:</span>{' '}
                    {hoveredNode.filePath}
                  </p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Graph Canvas */}
        <Card className="col-span-9 overflow-hidden">
          <CardContent className="p-0 h-full">
            {!selectedEntity ? (
              <div className="flex flex-col items-center justify-center h-full gap-4">
                <p className="text-muted-foreground">
                  Select a function from the Explorer to view its call graph
                </p>
                <Button variant="outline" onClick={() => navigate('/explorer')}>
                  Go to Explorer
                </Button>
              </div>
            ) : isLoadingGraph ? (
              <div className="flex items-center justify-center h-full">
                <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : !callGraph || callGraph.nodes.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full gap-4">
                <p className="text-muted-foreground">
                  No call graph data available for this entity
                </p>
              </div>
            ) : (
              <svg
                ref={svgRef}
                className="w-full h-full"
                style={{ minHeight: '500px' }}
              />
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
