package com.example;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.math.BigDecimal;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class OrderServiceTest {
    private OrderService orderService;

    @BeforeEach
    void setUp() {
        orderService = new OrderService();
    }

    @Test
    void createOrder_ValidItems_Success() {
        List<OrderItem> items = List.of(
            new OrderItem("Item1", BigDecimal.valueOf(10.00)),
            new OrderItem("Item2", BigDecimal.valueOf(20.00))
        );

        Order order = orderService.createOrder(items);

        assertNotNull(order.getId());
        assertEquals(BigDecimal.valueOf(30.00), order.getTotal());
        assertEquals(OrderStatus.PENDING, order.getStatus());
    }

    @Test
    void createOrder_EmptyItems_ThrowsException() {
        assertThrows(IllegalArgumentException.class, 
            () -> orderService.createOrder(List.of()));
    }

    @Test
    void updateOrderStatus_ExistingOrder_Success() {
        List<OrderItem> items = List.of(
            new OrderItem("Item1", BigDecimal.valueOf(10.00))
        );
        Order order = orderService.createOrder(items);

        Order updated = orderService.updateOrderStatus(
            order.getId(), 
            OrderStatus.COMPLETED
        );

        assertEquals(OrderStatus.COMPLETED, updated.getStatus());
    }
}